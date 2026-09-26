"""Benign frozen-transfer and router-factory contracts; no model loading."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from src.infra.artifacts import make_winner, skill_hash
from src.eval.router_transfer import evaluate_router_transfer
from src.routers.registry import build_router

CL = 'coding-and-devops'
TASKS = [{'task_id':'text-summary','instruction_text':'Summarize the supplied greeting.'}]
SKILL = {'name':'Summary', 'description':'Summarize text', 'body':'Read and summarize the text.'}


def winner(router='source-router'):
    return make_winner(stage='A',skill=SKILL,cluster=CL,router={'router':router},config={'seed':1},
        task_ids=['text-summary'],source_run='benign-fixture',score={'hit_at_1':1,'optimization_reward':1})


class Router:
    def __init__(self, rank=0): self.result=rank; self.calls=[]
    def prepare_pool(self,pool): self.pool=pool
    def provenance(self): return {'router':f'dummy-{self.result}'}
    def rank(self,query,skill):
        self.calls.append((query,deepcopy(skill)))
        return self.result
    def retrieval_similarity(self,query,skill): return 0.25


class RouterContracts(unittest.TestCase):
    def test_five_factories_and_retrieve_rerank_pairs(self):
        definitions = {
            'bm25':['BM25Router'], 'dense':['DenseEmbeddingRouter'],
            'openai_embeddings':['OpenAIEmbedder'], '_skillrouter_client':['SkillRouterClient'],
            'skillrouter':['SkillRouterFull'], 'r3':['R3SkillClient','R3SkillRouter'],
            'local_embeddings':['Qwen3Embedder'], 'native_rerankers':['Qwen3Reranker'],
            'retrieve_then_rerank':['DenseRetrieveThenRerankRouter']}
        modules={}; constructors={}
        for name, classes in definitions.items():
            module=ModuleType('src.routers.'+name)
            for cls in classes:
                constructor=Mock(name=cls); setattr(module,cls,constructor); constructors[cls]=constructor
            modules[module.__name__]=module
        modules['src.routers.r3'].R3_ENCODER='tencent/R3-embedding-0.6b'
        modules['src.routers.r3'].R3_RERANKER='tencent/R3-rerank-0.6b'
        with patch.dict(sys.modules, modules):
            for router in ('bm25','openai_embedding_3_large','full_skillrouter','r3_skill_06','qwen3_8b_pair'):
                build_router(router)
        constructors['BM25Router'].assert_called_once()
        self.assertEqual(constructors['OpenAIEmbedder'].call_args.args[0],'text-embedding-3-large')
        self.assertEqual(constructors['SkillRouterClient'].call_args.kwargs['reranker_path'],
                         'pipizhao/SkillRouter-Reranker-0.6B')
        constructors['SkillRouterFull'].assert_called_once()
        self.assertEqual(constructors['R3SkillClient'].call_args.kwargs['reranker_path'],'tencent/R3-rerank-0.6b')
        constructors['R3SkillRouter'].assert_called_once()
        self.assertEqual(constructors['Qwen3Embedder'].call_args.args[0],'Qwen/Qwen3-Embedding-8B')
        self.assertEqual(constructors['Qwen3Reranker'].call_args.args[0],'Qwen/Qwen3-Reranker-8B')
        constructors['DenseRetrieveThenRerankRouter'].assert_called_once()

    def test_frozen_matrix_same_scope_hashes_and_no_execution_or_optimization(self):
        from src.attackers.gepa_cluster_pareto import GepaClusterParetoAttacker
        sources={'first':[winner()], 'second':[winner('other-source')]}; before=deepcopy(sources)
        targets={'hit':Router(0),'miss':Router(1)}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(GepaClusterParetoAttacker,'optimize',side_effect=AssertionError('optimizer called')), \
             patch('src.exec_env.experiment.load_experiment',side_effect=AssertionError('execution configured')):
            result=evaluate_router_transfer(sources=sources,targets=targets,tasks=TASKS,
                categories={'text-summary':CL},pool=[{'skill_id':'reference','name':'Reference','description':'','body':'Text'}],
                out_dir=Path(tmp)/'transfer')
            rows=[json.loads(l) for l in (Path(tmp)/'transfer/rows.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),4)
            self.assertTrue(all(r['skill_sha256']==skill_hash(SKILL) for r in rows))
            self.assertTrue(all(r['asr'] is None and r['exec'] is None for r in rows))
            self.assertTrue((Path(tmp)/'transfer/hit_at_1_macro.csv').exists())
        self.assertEqual(sources,before)
        self.assertFalse(result['optimization_enabled'])
        for source in sources:
            self.assertEqual(result['cells'][source]['hit']['macro']['hit_at_1'],1)
            self.assertEqual(result['cells'][source]['miss']['macro']['hit_at_1'],0)
        self.assertEqual([q for q,_ in targets['hit'].calls],[q for q,_ in targets['miss'].calls])

    def test_scope_mismatch_fails_before_routing(self):
        from src.eval.router_transfer import validate_comparability
        with self.assertRaisesRegex(ValueError,'scope'):
            validate_comparability({'source':[winner()]},[{'task_id':'different'}],{'different':CL})

    def test_router_mutation_cannot_change_frozen_input(self):
        class MutatingRouter(Router):
            def rank(self,query,skill): skill['body']='Changed disposable copy'; return 0
        a=winner(); before=deepcopy(a)
        with tempfile.TemporaryDirectory() as tmp:
            evaluate_router_transfer(sources={'source':[a]},targets={'target':MutatingRouter()},tasks=TASKS,
                categories={'text-summary':CL},pool=[],out_dir=Path(tmp)/'out')
        self.assertEqual(a,before)

    def test_retrieval_cli_has_no_evaluator_requirement(self):
        from experiments.router_transfer.run import parse_args
        args=parse_args(['--winner','unused','--target-router','bm25','--out-dir','unused'])
        self.assertFalse(hasattr(args,'experiment'))

    def test_real_transfer_requires_execution_contract(self):
        from experiments.real_eval_ourmethod.run import main
        from src.exec_env.contracts import EvaluatorUnavailable
        with patch('experiments.real_eval_ourmethod.run.load_experiment',side_effect=EvaluatorUnavailable('required')) as load, \
             patch('experiments.real_eval_ourmethod.run.build_router_from_args') as router:
            with self.assertRaises(EvaluatorUnavailable):
                main(['--winner','unused','--experiment','experiment.json','--model','test',
                      '--scaffold','test','--out-dir','unused'])
            load.assert_called_once_with('experiment.json'); router.assert_not_called()

    def test_bm25_competition(self):
        from src.routers.bm25 import BM25Router
        from src.infra.artifacts import validate_skill
        r=BM25Router();r.prepare_pool([{'skill_id':'reference','name':'Weather','description':'Forecast','body':'Rain'}])
        self.assertEqual(r.rank('summarize text',validate_skill(SKILL)),0)


if __name__ == '__main__': unittest.main()
