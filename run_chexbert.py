from radmetric.chexbert.f1chexbert_worker import F1CheXbertRewardServer

server = F1CheXbertRewardServer()
server.compute_rewards(['test', 'test'], ['test', 'test'],)
server.compute_metrics(['test', 'test'], ['test', 'test'],)
