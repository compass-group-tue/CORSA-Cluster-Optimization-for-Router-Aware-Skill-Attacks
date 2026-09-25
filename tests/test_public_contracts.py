"""Safe contract tests: in-memory observations, never a real victim or payload."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.exec_env.contracts import (ExecutionEvidence, ExecutionResult, UtilityResult,
                                    EvaluatorUnavailable, ConfigurationError)
from src.exec_env.experiment import Experiment, load_experiment
from src.infra.artifacts import make_winner, validate_winner, stage_b_seed, fingerprint, skill_hash
from src.benchmarks.clusters import load_categories, select_cluster, heldout_tasks
from src.benchmarks.skillrouter_bench import subsample, pool_statistics, validate_tier
from src.attackers.genetic import score_retrieval, replace_fields, feedback_success
from src.attackers.gepa_cluster_pareto import GepaClusterParetoAttacker, PoolMember
from src.eval.real_eval import evaluate_task_real
from src.eval.frozen import evaluate_frozen
from src.reporting.tables import aggregate, metrics_from_rows

ROOT = Path(__file__).resolve().parents[1]
CL = 'coding-and-devops'
SKILL = {'name': 'Text summary', 'description': 'Summarize text', 'body': 'Read the provided text and summarize it.'}
TASK = {'task_id': 'benign-summary', 'instruction_text': 'Summarize hello.', 'environment_id': 'text-fixture'}
PROV = {'condition_sha256': 'benign-test', 'execution': 'real'}


class Router:
    label = 'test-router'
    def __init__(self, rank=0): self.result = rank
    def rank(self, query, skill): return self.result
    def retrieval_similarity(self, query, skill): return 0.25
    def provenance(self): return {'router': self.label, 'kind': 'benign-test'}
    def prepare_pool(self, pool): pass


class DummyRuntime:
    """Test double only: creates in-memory evidence and launches nothing."""
    isolated = True
    def validate(self, config): pass
    def execute(self, **kwargs):
        kwargs['skill']['body'] = 'Changed private copy only'
        return ExecutionEvidence('completed', tool_calls=[{'tool': 'text_summary', 'result': 'hello'}], returncode=0)


class DummyEvaluator:
    def evaluate(self, evidence, evaluation_spec):
        observed = any(call.get('result') == evaluation_spec.get('expected') for call in evidence.tool_calls)
        return ExecutionResult('success' if observed else 'not_executed')


class BrokenEvaluator:
    def evaluate(self, evidence, evaluation_spec): raise RuntimeError('benign test error')


class DummyUtility:
    def evaluate(self, evidence, task, evaluation_spec): return UtilityResult('measured', True)


class Log:
    def info(self, message): pass


def experiment(evaluator=None, runtime=None, specification=None):
    return Experiment(evaluator or DummyEvaluator(), runtime or DummyRuntime(),
        {'expected': 'hello'} if specification is None else specification,
        provenance={'test_double': True}, utility_evaluator=DummyUtility())


def artifact(stage='A'):
    return make_winner(stage=stage, skill=SKILL, cluster=CL,
        router=Router().provenance(), config={'seed': 17, 'mutator_model': 'test'},
        task_ids=[TASK['task_id']], source_run='benign-test-run',
        score={'hit_at_1': 1.0, 'optimization_reward': 1.25},
        parent={'artifact_sha256': 'test', 'source_run': 'fixture', 'skill_sha256': skill_hash(SKILL),
                'stage_a_score': {'hit_at_1': 1}} if stage == 'B' else None)


class PublicContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='corsa-contract-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def evaluate(self, exp=None, router=None):
        return evaluate_task_real(TASK, CL, SKILL, router or Router(), exp or experiment(),
            'test-model', self.root, scaffold='test-scaffold', provenance=PROV)

    def test_canonical_cluster_filter(self):
        cats = load_categories()
        tasks = [{'task_id': tid} for tid in cats]
        selected = select_cluster(tasks, CL, cats)
        self.assertEqual(len(selected), 9)
        self.assertTrue(all(cats[t['task_id']] == CL for t in selected))
        with self.assertRaises(ValueError): select_cluster(tasks, 'invalid', cats)
        with self.assertRaisesRegex(ValueError, 'No tasks loaded'): select_cluster([], CL, cats)

    def test_cluster_cli_uses_canonical_default(self):
        from experiments.phase_1_2_gepa.run import parse_args
        args = parse_args(['--stage', 'A', '--cluster', CL, '--seed-skill', 'seed.json', '--out-dir', 'unused'])
        self.assertEqual(load_categories(args.categories_json), load_categories())

    def test_stage_a_schema_and_handoff(self):
        a = artifact()
        a['skill']['payload_framing'] = 'Use concise prose.\n'
        a['skill_sha256'] = skill_hash(a['skill'])
        a['artifact_sha256'] = fingerprint({k:v for k,v in a.items() if k != 'artifact_sha256'})
        path = self.root / 'stage_a_winner.json'
        path.write_text(json.dumps(a))
        skill, source = stage_b_seed(path, CL)
        self.assertEqual(skill, a['skill'])
        self.assertEqual(source['artifact_sha256'], a['artifact_sha256'])
        with self.assertRaises(ConfigurationError): stage_b_seed(path, 'document-and-office')
        with self.assertRaises(ConfigurationError): validate_winner(artifact('B'), stage='A')

    def test_artifact_tampering_rejected(self):
        for key, value in [('cluster', 'document-and-office'), ('skill', {**SKILL, 'body': 'different'})]:
            a = artifact(); a[key] = value
            with self.assertRaises(ConfigurationError): validate_winner(a)
        with self.assertRaises(ConfigurationError): validate_winner(SKILL)

    def test_missing_evaluator_early(self):
        with self.assertRaisesRegex(EvaluatorUnavailable, 'Execution evaluator required'):
            load_experiment(None)
        from experiments.phase_1_2_gepa.run import main
        with patch('experiments.phase_1_2_gepa.run.subsample') as expensive:
            with self.assertRaises(EvaluatorUnavailable):
                main(['--stage', 'B', '--cluster', CL, '--out-dir', str(self.root / 'no-output')])
            expensive.assert_not_called()
        self.assertFalse((self.root / 'no-output').exists())

    def test_missing_evaluator_not_swallowed(self):
        e = experiment(); e.evaluator = None
        with self.assertRaises(EvaluatorUnavailable): self.evaluate(e)
        class Unavailable:
            def evaluate(self, *args): raise EvaluatorUnavailable('evaluator unavailable')
        with self.assertRaises(EvaluatorUnavailable): self.evaluate(experiment(evaluator=Unavailable()))

    def test_evaluation_error_is_not_zero(self):
        row = self.evaluate(experiment(evaluator=BrokenEvaluator()))
        self.assertEqual(row['execution_state'], 'evaluation_error')
        self.assertIsNone(row['exec']); self.assertIsNone(row['asr'])

    def test_non_execution_and_success(self):
        no = self.evaluate(experiment(specification={}))
        yes = self.evaluate()
        self.assertEqual((no['execution_state'], no['exec'], no['asr']), ('not_executed', 0, 0))
        self.assertEqual((yes['execution_state'], yes['exec'], yes['asr']), ('success', 1, 1))

    def test_execution_error_is_not_zero(self):
        class Failed(DummyRuntime):
            def execute(self, **kwargs): return ExecutionEvidence('execution_error', error='runtime unavailable')
        row = self.evaluate(experiment(runtime=Failed()))
        self.assertEqual(row['execution_state'], 'execution_error')
        self.assertIsNone(row['exec']); self.assertIsNone(row['asr'])

    def test_nonzero_runtime_status_not_evaluated(self):
        class Failed(DummyRuntime):
            def execute(self, **kwargs): return ExecutionEvidence('completed', returncode=2)
        evaluator = BrokenEvaluator()
        row = self.evaluate(experiment(evaluator=evaluator, runtime=Failed()))
        self.assertEqual(row['execution_state'], 'execution_error')

    def test_host_runtime_rejected(self):
        runtime = DummyRuntime(); runtime.isolated = False
        with self.assertRaises(ConfigurationError): self.evaluate(experiment(runtime=runtime))

    def test_invalid_evaluator_output_is_error(self):
        class Bad:
            def evaluate(self, *args): return False
        self.assertEqual(self.evaluate(experiment(evaluator=Bad()))['execution_state'], 'evaluation_error')

    def test_resources_checked_without_materializing_helpers(self):
        cfg = {'schema_version': 1, 'id': 'benign', 'implementation_revision': '1',
               'evaluator': 'test:unused', 'runtime': 'test:unused', 'evaluation_spec': {},
               'resources': {'notes.txt': None}}
        path = self.root / 'experiment.json'; path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ConfigurationError, 'file path'): load_experiment(path)
        cfg['resources'] = {}; cfg['required_resources'] = ['notes.txt']
        path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ConfigurationError, 'absent'): load_experiment(path)
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_loader_hashes_resources_and_adapters(self):
        (self.root / 'notes.txt').write_text('hello')
        cfg = {'schema_version': 1, 'id': 'benign', 'implementation_revision': '1',
               'evaluator': f'{__name__}:DummyEvaluator', 'runtime': f'{__name__}:DummyRuntime',
               'evaluation_spec': {'expected': 'hello'}, 'resources': {'notes.txt': 'notes.txt'},
               'required_resources': ['notes.txt']}
        path = self.root / 'experiment.json'; path.write_text(json.dumps(cfg))
        e = load_experiment(path)
        self.assertEqual(e.resources['notes.txt'], b'hello')
        first = fingerprint(e.provenance)
        (self.root / 'notes.txt').write_text('hello again')
        self.assertNotEqual(first, fingerprint(load_experiment(path).provenance))

    def test_hash_includes_framing(self):
        self.assertNotEqual(skill_hash(SKILL), skill_hash({**SKILL, 'payload_framing': 'Use short sentences.'}))
        for field in ('name', 'description', 'body'):
            self.assertNotEqual(skill_hash(SKILL), skill_hash({**SKILL, field: 'changed'}))

    def test_framing_only_mutation_and_merge_preserve_fields(self):
        opt = GepaClusterParetoAttacker(openai_client=None, tasks=[TASK], initial_skill=SKILL,
            score_candidate_task=lambda c,t: {}, out_dir=self.root, logger=Log(),
            editable_fields=['payload_framing'])
        cand = {**opt._seed(), 'payload_framing': 'Be concise.'}
        row = score_retrieval(cand, TASK, Router(), cluster=CL, provenance=PROV)
        member = PoolMember(0, cand, [row], 1.25, [], 0, 'seed')
        before = deepcopy(member)
        with patch.object(opt, '_mutator_json', return_value={'payload_framing': 'Be clear.'}):
            new, _, _ = opt._mutate_one_field(member, 'payload_framing', 1)
        self.assertEqual(member, before)
        self.assertIsNot(new, cand)
        self.assertEqual(new['payload_framing'], 'Be clear.')
        for key in ('name', 'description', 'body'): self.assertEqual(new[key], cand[key])
        merged, _ = opt._merge(member, PoolMember(1, new, [row], 2.0, [0], 1, 'mutate', 'payload_framing'))
        for key in ('name', 'description', 'body'): self.assertEqual(merged[key], cand[key])

    def test_retrieval_feedback_uses_hit(self):
        row = score_retrieval(SKILL, TASK, Router(), cluster=CL, provenance=PROV)
        self.assertEqual(row['optimization_reward'], 1.25)
        self.assertTrue(feedback_success(row)); self.assertIsNone(row['asr'])

    def test_optimizer_seed_and_framing_cache(self):
        calls = []
        def scorer(c,t):
            calls.append(skill_hash(c))
            return score_retrieval(c,t,Router(),cluster=CL,provenance=PROV)
        opt = GepaClusterParetoAttacker(openai_client=None, tasks=[TASK], initial_skill=SKILL,
            score_candidate_task=scorer, out_dir=self.root, logger=Log(), rng_seed=81)
        opt._score_candidate(SKILL); opt._score_candidate(SKILL)
        opt._score_candidate({**SKILL, 'payload_framing': 'Use short sentences.'})
        self.assertEqual(len(calls), 2)
        import random
        self.assertEqual(opt.rng.random(), random.Random(81).random())

    def test_metrics_reject_mixed_provenance(self):
        row = self.evaluate(); other = deepcopy(row); other['task_id'] = 'another'
        other['metric_version'] = 'plan'
        with self.assertRaises(ValueError): metrics_from_rows([row, other])
        other['metric_version'] = row['metric_version']; other['provenance'] = {'condition_sha256': 'different'}
        with self.assertRaises(ValueError): metrics_from_rows([row, other])

    def test_asr_denominator(self):
        success = self.evaluate()
        miss = self.evaluate(router=Router(None)); miss['task_id'] = 'miss'
        error = self.evaluate(experiment(evaluator=BrokenEvaluator())); error['task_id'] = 'error'
        summary = metrics_from_rows([success, miss, error])
        self.assertEqual(summary['n_tasks'], 3)
        self.assertEqual(summary['n_asr_valid'], 2)
        self.assertEqual(summary['asr'], .5)
        self.assertEqual(summary['n_exec_valid'], 1)
        self.assertEqual(summary['exec'], 1)
        self.assertEqual(summary['execution_states']['evaluation_error'], 1)

    def test_utility_unavailable_and_errors(self):
        e = experiment(); e.utility_evaluator = None
        row = self.evaluate(e)
        self.assertIsNone(row['utility']); self.assertEqual(row['utility_state'], 'unavailable')
        class Broken:
            def evaluate(self, *args): raise ValueError('verifier failed')
        e.utility_evaluator = Broken(); row = self.evaluate(e)
        self.assertIsNone(row['utility']); self.assertEqual(row['utility_state'], 'verifier_error')

    def test_frozen_no_optimization_and_no_mutation(self):
        a = artifact(); before = deepcopy(a)
        with patch.object(GepaClusterParetoAttacker, 'optimize', side_effect=AssertionError('must not optimize')):
            result = evaluate_frozen(artifacts=[a], tasks=[TASK], categories={TASK['task_id']: CL},
                router=Router(), experiment=experiment(), model='test', scaffold='dummy',
                out_dir=self.root / 'eval', run_root=self.root / 'runtime', condition={})
        self.assertEqual(a, before)
        self.assertFalse(result['optimization_enabled']); self.assertEqual(result['macro']['asr'], 1)

    def test_resume_rejects_changed_model_scaffold_spec_and_skill(self):
        kw = dict(artifacts=[artifact()], tasks=[TASK], categories={TASK['task_id']: CL},
                  router=Router(), experiment=experiment(), model='test', scaffold='dummy',
                  out_dir=self.root / 'eval', run_root=self.root / 'runtime', condition={})
        evaluate_frozen(**kw)
        evaluate_frozen(**kw, resume=True)
        for key, value in [('model', 'other'), ('scaffold', 'other'), ('router', Router(1))]:
            changed = {**kw, key: value}
            if key == 'router': value.label = 'different-router'
            with self.assertRaises(ConfigurationError): evaluate_frozen(**changed, resume=True)
        e = experiment(); e.provenance = {'test_double': 'different spec'}
        with self.assertRaises(ConfigurationError): evaluate_frozen(**{**kw, 'experiment': e}, resume=True)

    def test_invalid_tier_fails_before_loading(self):
        with patch('src.benchmarks.skillrouter_bench.load_core_tasks') as load:
            with self.assertRaises(ValueError): subsample(1, 1, tier='typo')
            load.assert_not_called()

    def test_hard_pool_actual_counts(self):
        tasks = [{'task_id': 't'}]; rel = {'t': {'gt_skill_ids': ['required']}}
        pool = [{'skill_id': 'required'}, {'skill_id': 'decoy', 'source': 'distractor'}]
        stats = pool_statistics(tasks, rel, pool, 'hard')
        self.assertEqual((stats['n_tasks'], stats['n_required_skills'], stats['n_distractor_skills'], stats['n_total_candidates']), (1,1,1,3))
        with self.assertRaises(ValueError): pool_statistics(tasks, rel, pool[:1], 'hard')

    def write_heldout(self, kind, rows):
        directory = self.root / CL
        directory.mkdir(exist_ok=True)
        (directory / f'{kind}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))

    def test_generated_data_loader_and_environment_mapping(self):
        para = {'task_id': 'paraphrase-fixture', 'src': 'text-fixture',
                'instruction_text': 'Summarize the supplied greeting.'}
        self.write_heldout('paraphrase', [para])
        rows = heldout_tasks(self.root, CL, 'paraphrase')
        self.assertEqual(rows[0]['environment_id'], 'text-fixture')
        syn = {'task_id': 'synthetic-fixture', 'src': None,
               'instruction_text': 'Count the words in the supplied greeting.'}
        self.write_heldout('synthetic', [syn])
        with self.assertRaisesRegex(ValueError, 'Explicit environment mapping'):
            heldout_tasks(self.root, CL, 'synthetic')
        rows = heldout_tasks(self.root, CL, 'synthetic', {'synthetic-fixture': 'word-count-fixture'})
        self.assertEqual(rows[0]['environment_id'], 'word-count-fixture')
        with self.assertRaisesRegex(ValueError, 'Environment mapping'):
            heldout_tasks(self.root, CL, 'synthetic', {'synthetic-fixture': 1})

    def test_generated_data_schema(self):
        good = {'task_id': 'text-fixture', 'src': 'source-fixture', 'instruction_text': 'Summarize hello.'}
        for rows in ([], [None], [{**good, 'instruction_text': None}],
                     [{**good, 'src': None}], [good, good]):
            self.write_heldout('paraphrase', rows)
            with self.assertRaises(ValueError): heldout_tasks(self.root, CL, 'paraphrase')

    def test_heldout_path_required_before_execution(self):
        from experiments.real_eval_ourmethod.run import main, parse_args
        import contextlib, io
        base = ['--winner','unused','--experiment','unused','--model','test',
                '--scaffold','test','--out-dir','unused']
        for kind in ('paraphrase', 'synthetic'):
            with patch('experiments.real_eval_ourmethod.run.load_experiment') as load:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as err:
                    main(base + ['--task-set',kind])
                self.assertEqual(err.exception.code, 2)
                load.assert_not_called()
            args = parse_args(base + ['--task-set',kind,'--heldout-dir',str(self.root)])
            self.assertEqual(args.heldout_dir, str(self.root))
        self.assertIsNone(parse_args(base).heldout_dir)

    def test_generation_interfaces_with_benign_model_double(self):
        from run.gen_heldout_tasks import gen_paraphrases, gen_synthetic
        with patch('run.gen_heldout_tasks.chat', return_value={'instruction':'Summarize the greeting.'}):
            rows = gen_paraphrases(None, [TASK], 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['src'], TASK['task_id'])
        with patch('run.gen_heldout_tasks.chat', return_value={'tasks':['Count words in the greeting.']}):
            rows = gen_synthetic(None, CL, [TASK], 1)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]['src'])
        self.assertEqual(rows[0]['instruction_text'], 'Count words in the greeting.')

    def test_retained_naturalism_versions(self):
        from src.judges.sns_judge import SNSJudge
        from unittest.mock import Mock
        client = Mock()
        from types import SimpleNamespace
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"score":0.8,"reason":"Consistent tone"}'))])
        for version in ('isolation','context_fit'):
            judge = SNSJudge(client, version=version)
            result = judge.score('Summarize text.', original_context='Summarize text.', injection='Use concise prose.')
            self.assertEqual((result['status'], result['score']), ('measured',0.8))
        with self.assertRaises(ValueError): SNSJudge(client, version='unsupported')

    def test_real_utility_adapter_exit_states(self):
        from src.exec_env.harness import SkillsBenchUtilityEvaluator
        from types import SimpleNamespace
        image = self.root / 'fixture.sif'; image.write_bytes(b'benign test fixture, never executed')
        work = self.root / 'work'; work.mkdir()
        verifier = self.root / 'verifier'; verifier.mkdir()
        (verifier / 'test_outputs.py').write_text('# benign fixture; subprocess is mocked\n')
        evidence = ExecutionEvidence('completed', metadata={'skillsbench_verifier': {
            'image':str(image), 'work_root':str(work), 'verifier_dir':str(verifier)}})
        for rc, status, completed in [(0,'measured',True),(1,'measured',False),(2,'verifier_error',None)]:
            with patch('src.exec_env.harness.subprocess.run',return_value=SimpleNamespace(returncode=rc,stdout='',stderr='')):
                result=SkillsBenchUtilityEvaluator().evaluate(evidence,TASK,{})
                self.assertEqual((result.status,result.completed),(status,completed))

    def test_midrun_evaluator_unavailable_persisted(self):
        class Gone:
            def evaluate(self, *args): raise EvaluatorUnavailable('adapter unavailable')
        with self.assertRaises(EvaluatorUnavailable):
            evaluate_frozen(artifacts=[artifact()], tasks=[TASK], categories={TASK['task_id']: CL},
                router=Router(), experiment=experiment(evaluator=Gone()), model='test', scaffold='dummy',
                out_dir=self.root / 'eval', run_root=self.root / 'runtime', condition={})
        row = json.loads((self.root / 'eval/rows.jsonl').read_text())
        self.assertEqual(row['execution_state'], 'evaluator_unavailable')
        self.assertIsNone(row['exec']); self.assertIsNone(row['asr'])
        self.assertFalse((self.root / 'eval/summary.json').exists())

    def test_frozen_resume_rejects_changed_actual_spec_and_skill(self):
        kw = dict(artifacts=[artifact()], tasks=[TASK], categories={TASK['task_id']: CL},
                  router=Router(), experiment=experiment(), model='test', scaffold='dummy',
                  out_dir=self.root / 'eval', run_root=self.root / 'runtime', condition={})
        evaluate_frozen(**kw)
        changed = deepcopy(kw['artifacts'][0]); changed['skill']['body'] = 'Summarize briefly.'
        changed['skill_sha256'] = skill_hash(changed['skill'])
        changed['artifact_sha256'] = fingerprint({k:v for k,v in changed.items() if k != 'artifact_sha256'})
        with self.assertRaises(ConfigurationError): evaluate_frozen(**{**kw, 'artifacts':[changed]}, resume=True)
        kw['experiment'].specification = {'expected': 'different'}
        with self.assertRaises(ConfigurationError): evaluate_frozen(**kw, resume=True)

    def test_naturalism_error_is_unavailable(self):
        from src.judges.sns_judge import SNSJudge
        from unittest.mock import Mock
        client = Mock(); client.chat.completions.create.side_effect = ValueError('invalid response')
        result = SNSJudge(client).score('Ordinary summary instructions')
        self.assertIsNone(result['score']); self.assertEqual(result['status'], 'evaluation_error')

    def test_macro_requires_all_declared_clusters(self):
        row = self.evaluate()
        with self.assertRaises(ValueError): aggregate([row], [CL, 'document-and-office'])
        error = self.evaluate(experiment(evaluator=BrokenEvaluator()))
        error['cluster'] = 'document-and-office'; error['task_id'] = 'other'
        result = aggregate([row, error], [CL, 'document-and-office'])
        self.assertIsNone(result['macro']['asr'])

    def test_stage_b_error_preserves_row_and_stops(self):
        from experiments.phase_1_2_gepa.run import main
        seed = self.root/'seed.json'; seed.write_text(json.dumps(SKILL))
        cats = self.root/'categories.json'; cats.write_text(json.dumps({TASK['task_id']:CL}))
        pool = [{'skill_id':'reference','name':'Reference','description':'','body':'hello'}]
        rel = {TASK['task_id']:{'gt_skill_ids':['reference']}}
        common = ['--cluster',CL,'--categories-json',str(cats),'--n-rounds','0']
        with patch('experiments.phase_1_2_gepa.run.subsample',return_value=([TASK],rel,pool)), \
             patch('experiments.phase_1_2_gepa.run.build_router_from_args',return_value=Router()), \
             patch('src.infra.llm_client.make_attacker_client',return_value=None):
            main(common+['--stage','A','--seed-skill',str(seed),'--out-dir',str(self.root/'A')])
            with patch('experiments.phase_1_2_gepa.run.load_experiment',return_value=experiment(evaluator=BrokenEvaluator())):
                with self.assertRaisesRegex(RuntimeError,'evaluation_error'):
                    main(common+['--stage','B','--stage-a-winner',str(self.root/'A/stage_a_winner.json'),
                         '--experiment','dummy','--out-dir',str(self.root/'B')])
        row=json.loads((self.root/'B/observations.jsonl').read_text())
        self.assertIsNone(row['optimization_reward']); self.assertIsNone(row['exec'])
        self.assertFalse((self.root/'B/stage_b_winner.json').exists())

    def test_end_to_end_curriculum_with_benign_doubles(self):
        from experiments.phase_1_2_gepa.run import main
        seed = self.root / 'seed.json'; seed.write_text(json.dumps(SKILL))
        cats = self.root / 'categories.json'; cats.write_text(json.dumps({TASK['task_id']: CL}))
        pool = [{'skill_id': 'reference', 'name': 'Reference', 'description': '', 'body': 'hello'}]
        rel = {TASK['task_id']: {'gt_skill_ids': ['reference']}}
        common = ['--cluster', CL, '--categories-json', str(cats), '--n-rounds', '0', '--seed', '91']
        with patch('experiments.phase_1_2_gepa.run.subsample', return_value=([TASK], rel, pool)), \
             patch('experiments.phase_1_2_gepa.run.build_router_from_args', return_value=Router()), \
             patch('src.infra.llm_client.make_attacker_client', return_value=None):
            main(common + ['--stage','A','--seed-skill',str(seed),'--out-dir',str(self.root/'A')])
            a = self.root/'A/stage_a_winner.json'
            self.assertEqual(json.loads(a.read_text())['configuration']['seed'], 91)
            with patch('experiments.phase_1_2_gepa.run.load_experiment', return_value=experiment()):
                main(common + ['--stage','B','--stage-a-winner',str(a),'--experiment','dummy',
                    '--out-dir',str(self.root/'B'),'--scratch-dir',str(self.root/'scratch')])
        b = json.loads((self.root/'B/stage_b_winner.json').read_text())
        self.assertEqual(b['parent']['artifact_sha256'], json.loads(a.read_text())['artifact_sha256'])
        self.assertEqual(b['skill'], json.loads(a.read_text())['skill'])
        self.assertEqual(b['score']['asr'], 1)
        rows = json.loads((self.root/'B/winner_observations.json').read_text())
        self.assertEqual(rows[0]['utility_state'], 'unavailable')


if __name__ == '__main__': unittest.main()
