"""Package-scanner contracts with harmless text fixtures and mocked processes."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.defenses.packages import package_files, package_hash, copy_full_package
from src.defenses.scanner import ScannerConfig, scanner_env
from src.defenses import cisco, skillspector
from src.defenses.metrics import aggregate_scans
from src.defenses.scanner_transport import adapt_request


def cisco_report(severity=None,llm=False):
    return {'analyzers_used':sorted(cisco.CORE_ANALYZERS)+(['llm_analyzer'] if llm else []),
            'findings':[{'severity':severity}] if severity else [],
            'llm_usage':{'total_tokens':4} if llm else None}


def nvidia_report(score,llm=False):
    meta={'llm_requested':llm}
    if llm: meta.update(llm_available=True,llm_degraded=False,llm_calls_attempted=1,
        llm_calls_succeeded=1,inference_usage=[{'usage_source':'provider_response','total_tokens':4}])
    return {'execution_successful':True,'metadata':meta,'risk_assessment':{'score':score},'issues':[]}


class ScannerContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.package=self.root/'package';self.package.mkdir()
        (self.package/'SKILL.md').write_text('---\nname: text-summary\ndescription: Summarize text\n---\nRead the supplied text.')
        (self.package/'notes.txt').write_text('Harmless reference text.')

    def test_full_package_hash_and_copy(self):
        digest=package_hash(self.package)
        copied,files=copy_full_package(self.package,self.root/'staged',digest)
        self.assertEqual(files,['SKILL.md','notes.txt']);self.assertEqual(package_hash(copied),digest)
        (self.package/'notes.txt').write_text('Different reference text')
        self.assertNotEqual(package_hash(self.package),digest)
        with self.assertRaises(ValueError):copy_full_package(self.package,self.root/'changed',digest)

    def test_symlinks_and_nested_packages_rejected(self):
        link=self.package/'link.txt';link.symlink_to(self.package/'notes.txt')
        with self.assertRaises(ValueError):package_files(self.package)
        link.unlink();rootlink=self.root/'linked';rootlink.symlink_to(self.package)
        with self.assertRaises(ValueError):package_files(rootlink)
        nested=self.package/'nested';nested.mkdir();(nested/'SKILL.md').write_text('Benign nested text')
        with self.assertRaises(ValueError):package_files(self.package)

    def test_cisco_high_critical_only(self):
        config=ScannerConfig('cisco')
        for severity in ('NONE','INFO','LOW','MEDIUM','HIGH','CRITICAL'):
            _,verdict=cisco.normalize_report(cisco_report(severity),config)
            self.assertEqual(verdict,'malicious' if severity in ('HIGH','CRITICAL') else 'benign')
        with self.assertRaises(ValueError):cisco.normalize_report(cisco_report('unknown'),config)

    def test_skillspector_boundary_invalid_scores(self):
        config=ScannerConfig('skillspector')
        for score in (0,49,50,50.01,100):
            self.assertEqual(skillspector.normalize_report(nvidia_report(score),config)[1],
                             'malicious' if score>50 else 'benign')
        for score in (True,None,float('nan'),float('inf'),-1,101):
            with self.assertRaises(ValueError):skillspector.normalize_report(nvidia_report(score),config)

    def test_invalid_denominators_recall_fpr(self):
        rows=[]
        for label,verdict,status in [('malicious','malicious','valid'),('malicious',None,'invalid'),
                                     ('benign','malicious','valid'),('benign',None,'invalid')]:
            rows.append(dict(sample_id=str(len(rows)),label=label,verdict=verdict,status=status,config={'scanner':'test'}))
        result=aggregate_scans(rows)
        self.assertEqual((result['recall'],result['fpr'],result['n_evaluated'],result['n_invalid']),(.5,.5,4,2))
        self.assertEqual(result['false_negatives'],1)
        self.assertEqual(result['by_label']['benign']['n_flagged'],1)
        rows[0]['config']={'scanner':'other'}
        with self.assertRaises(ValueError):aggregate_scans(rows)

    def test_configuration_static_llm_dispatch_and_profiles(self):
        for module,name in ((cisco,'cisco'),(skillspector,'skillspector')):
            for profile in ('static','qwen3.8-27b','gpt-5.4'):
                config=ScannerConfig(name,profile,base_url='http://localhost:1234/v1')
                with patch.dict('os.environ',{'OPENAI_API_KEY':'test-only'}):
                    config.validate();env=scanner_env(config,config.base_url,self.root)
                args=module.command('scanner',self.package,self.root/'output.json',config)
                if name=='cisco':self.assertEqual('--use-llm' in args,profile!='static')
                else:self.assertEqual('--no-llm' in args,profile=='static')
                if profile=='static':self.assertNotIn('OPENAI_API_KEY',env)
                elif name=='skillspector':self.assertTrue(Path(env['SKILLSPECTOR_MODEL_REGISTRY']).is_file())
        with self.assertRaises(ValueError):ScannerConfig('cisco','unknown').validate()
        with self.assertRaises(ValueError):ScannerConfig('cisco','qwen3.8-27b').validate()
        with self.assertRaises(ValueError):ScannerConfig('skillspector',schema_compat=True).validate()

    def test_llm_reports_require_mode_and_usage(self):
        for module,name,report in [(cisco,'cisco',cisco_report('LOW',True)),
                                   (skillspector,'skillspector',nvidia_report(20,True))]:
            config=ScannerConfig(name,'gpt-5.4')
            self.assertEqual(module.normalize_report(report,config)[1],'benign')
            with self.assertRaises(ValueError):module.normalize_report(report,ScannerConfig(name))
        bad=nvidia_report(20,True);bad['metadata']['llm_calls_succeeded']=0
        with self.assertRaises(ValueError):skillspector.normalize_report(bad,ScannerConfig('skillspector','gpt-5.4'))

    def test_native_invocation_full_package_and_no_labels(self):
        for module,name,version,report in [(cisco,'cisco','2.1.0',cisco_report('HIGH')),
                                           (skillspector,'skillspector','2.11.2',nvidia_report(51))]:
            calls=[]
            def process(args,**kw):
                calls.append(args)
                if args[-1]=='--version':return SimpleNamespace(returncode=0,stdout=version,stderr='')
                staged=Path(args[2]);self.assertEqual(package_hash(staged),package_hash(self.package))
                flag='--output-json' if name=='cisco' else '--output'
                Path(args[args.index(flag)+1]).write_text(json.dumps(report))
                return SimpleNamespace(returncode=0 if name=='cisco' else 1,stdout='',stderr='')
            with patch('src.defenses.scanner.subprocess.run',side_effect=process):
                result=module.scan_package(self.package,package_hash(self.package),ScannerConfig(name,executable='mock-scanner'))
            self.assertEqual((result['status'],result['verdict']),('valid','malicious'))
            self.assertEqual(len(calls),2);self.assertNotIn('label',result)

    def test_invalid_timeout_and_version_have_no_retry(self):
        import subprocess
        for failure in (subprocess.TimeoutExpired('scanner',1),ValueError('mock error')):
            with patch('src.defenses.scanner.subprocess.run',side_effect=failure) as proc:
                result=cisco.scan_package(self.package,package_hash(self.package),ScannerConfig('cisco',executable='mock'))
            self.assertEqual(result['status'],'invalid');proc.assert_called_once()
        with patch('src.defenses.scanner.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='0.0.0',stderr='')):
            result=cisco.scan_package(self.package,package_hash(self.package),ScannerConfig('cisco',executable='mock'))
        self.assertEqual(result['status'],'invalid')

    def test_narrow_transport_preserves_other_constraints_and_input(self):
        payload={'messages':[{'role':'user','content':'Review this benign text.'}],
                 'response_format':{'type':'json_schema','json_schema':{'schema':{'properties':{
                    'evidence_ids':{'type':'array','uniqueItems':True,'minItems':1},
                    'other':{'type':'array','uniqueItems':True}}}}}}
        before=deepcopy(payload)
        result=adapt_request(payload,disable_thinking=True,evidence_ids_compat=True)
        fields=result['response_format']['json_schema']['schema']['properties']
        self.assertNotIn('uniqueItems',fields['evidence_ids'])
        self.assertEqual(fields['evidence_ids']['minItems'],1)
        self.assertTrue(fields['other']['uniqueItems'])
        self.assertEqual(result['messages'],payload['messages']);self.assertEqual(payload,before)
        self.assertFalse(result['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(adapt_request(payload),payload)

    def test_runner_labels_stay_outside_adapter(self):
        from experiments.defenses.run import load_manifest,evaluate_packages
        manifest=self.root/'manifest.json'
        manifest.write_text(json.dumps([dict(sample_id='benign-fixture',package_path='package',label='benign')]))
        samples=load_manifest(manifest)
        with patch('experiments.defenses.run.cisco.scan_package',return_value=dict(status='invalid',verdict=None,
                   score=None,config={'scanner':'test'})) as scan:
            result=evaluate_packages(samples,ScannerConfig('cisco'),self.root/'out')
        self.assertEqual(len(scan.call_args.args),3)
        self.assertEqual(result['fpr'],0);self.assertEqual(result['n_invalid'],1)


if __name__ == '__main__': unittest.main()
