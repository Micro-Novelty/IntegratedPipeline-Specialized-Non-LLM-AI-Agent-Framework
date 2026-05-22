from AbstractIntegratedModule import CohesiveAgentDeployment
from AbstractIntegratedModule import PipelinePredictionManager
import asyncio
import traceback

prediction_manager = PipelinePredictionManager(main_model, label_csv=<your_training_labels.txt>, target_title=<target_title>, label=<target_label>)

print("=== SECURE PEER-TO-PEER CLUSTER ===")

# CohesiveAgentDeployment is deeply tied and coupled with AgentDistributedInference,
# if you already set an SSL cert and key, CohesiveAgentDeployment will use the SSL directly from AgentDistributedInference
# allowing secure socket to be used directly by CohesiveAgentDeployment

# Agent 1 - Primary (Port 5555)
agent1 = CohesiveAgentDeployment(
     memory_name="agent_primary",
     filename=<filename>,
     target_title=<title_name>,
     label_name=<label_name>,
     security_level="PRODUCTION",
     enable_peers=True,
     trusted_networks=['127.0.0.1/32', '192.168.1.0/24'], # for trusted networks, you need to provide the list of IPs of your peers.
     peer_discovery_port=5555, # peer port to start P2P
     secret_key=secret_key,
     shared_auth_token=secret_key, # your previous initialized secret_key
     predict_manager=prediction_manager,
     peer_config = <'your_peer_ip_lists.json'> # you need to create .json file that contains your peer IP and Port lists
     consecutive_peer_config = <'your_second_fallback_peer_ip_lists.json'> # same for this one too, but for fallback.
 )
 
# Agent 2 - Secondary (Port 5556)
agent2 = CohesiveAgentDeployment(
     memory_name="agent_secondary",
     filename=<filename>,
     target_title=<title_name>,
     label_name=<label_name>,
     security_level="PRODUCTION",
     enable_peers=True, # agent is allowed to find peers
     trusted_networks=['127.0.0.1/32', '192.168.1.0/24'],
     peer_discovery_port=5556,
     secret_key=secret_key,
     shared_auth_token=secret_key,
     predict_manager=prediction_manager,
     peer_config = <'your_peer_ip_lists.json'> # you need to create .json file that contains your peer IP and Port lists
     consecutive_peer_config = <'your_second_fallback_peer_ip_lists.json'> # same for this one too, but for fallback.
 )

# Note: CohesiveAgentDeployment contains ConsecutivePeerAgent that can start a server once ensemble prediction from peer is started
# be advised to stop the server too before shutdown-ing CohesiveAgentDeployment cluster

# example peer_Ip_lists_config.json (de-comment to use)
# {
      #  "known_peers": [ # you must put "known_peers" in the config so python can identify the list of IPs and Ports 
        #    ["127.0.0.1", 5555], can be modified using real IP or local IP.
        #    ["127.0.0.1", 5556]
     #   ]
    # }

agent1.pipeline = main_model # overrides agent1 baseline pipeline with your original initialized pipelinej
agent2.pipeline = main_model
 
try:
     # Start both agents
     print("\n🚀 Starting Agent 1...")
     await agent1.start()
     print("✅ Agent 1 started on port 5555")
     
     print("\n🚀 Starting Agent 2...")
     await agent2.start()
     print("✅ Agent 2 started on port 5556")
     
     # Give servers time to fully bind
     await asyncio.sleep(2)
     
     # Get API keys
     api_key = agent1.get_api_key()
     print(f"\n🔑 Using API Key: {api_key[:20]}...")
     
     # Connect peers (will stop on success)
     print("\n🔗 Connecting peers...")
     
     # Connect Agent 1 -> Agent 2 for testing
     success1 = await agent1._connect_with_smart_retry(agent1, "127.0.0.1", 5556, api_key) # can be changed with real IP
     if success1:
         print("✅ Agent 1 connected to Agent 2")
     else:
         print("❌ Agent 1 failed to connect to Agent 2")
     
     # Connect Agent 2 -> Agent 1
     success2 = await agent2._connect_with_smart_retry(agent2, "127.0.0.1", 5555, api_key)
     if success2:
         print("✅ Agent 2 connected to Agent 1")
     else:
         print("❌ Agent 2 failed to connect to Agent 1")
     
     # Wait for connections to establish
     await asyncio.sleep(1)
     
     # Display connection status
     print("\n📡 Connection Status:")
     print(f"   Agent 1 Peers: {len(agent1.list_peers())}")
     for peer in agent1.list_peers():
         print(f"      → {peer.get('host')}:{peer.get('port')}")
     
     print(f"   Agent 2 Peers: {len(agent2.list_peers())}")
     for peer in agent2.list_peers():
         print(f"      → {peer.get('host')}:{peer.get('port')}")
     
     # Only proceed if both agents have peers
     if len(agent1.list_peers()) == 0 or len(agent2.list_peers()) == 0:
         print("\n⚠️ Peers not fully connected. Skipping prediction test.")
     else:
         texts = {"test_titles": test_titles, "label_map": label_map, "rules": rules, "use_transformer": True, "agent_id": agent_id}

     # texts contains test_titles, label_map, and rules that you can assign,
     # agent ID can be strings, int, or floats, recommendeded to make it long for better security.

         # Make prediction with peer ensemble
         result = await agent1.multi_modal_peer_ensemble_prediction(
             texts=texts,
             api_key=api_key,
             method='advanced',
             disable_sync=True
         )  # await using asyncio, multi_modal_peer_ensemble is already async by design (Inside ConsecutivePeerAgent), no need to put asyncio.run() 

         result2 = await agent2.multi_modal_peer_ensemble_prediction(
             texts=texts,
             api_key=api_key,
             method='advanced',
             disable_sync=True
         )      
         
         print(f"\n📊 Ensemble Result for Agent 1:")
         print(f"   Prediction: {result.get('prediction', 'N/A')}")
         print(f"   Confidence: {result.get('confidence', 0):.2%}")

         print(f"\n📊 Ensemble Result for Agent 2:")
         print(f"   Prediction: {result2.get('prediction', 'N/A')}")
         print(f"   Confidence: {result2.get('confidence', 0):.2%}")

     # Keep running briefly
     print("\n⏳ Cluster stable. Waiting 5 seconds before shutdown...") # 5 seconds before shutdown.

     # stop ConsecutivePeerAgent servers inside CohesiveAgentDeployment.
     agent1._peer_agent.stop_server() # ._peer_agent is ConsecutivePeerAgent
     agent2._peer_agent.stop_server()

     await asyncio.sleep(5)

except Exception as e:
     print(f"\n❌ Error in cluster: {e}")
     traceback.print_exc()
     
finally:
     print("\n🛑 Shutting down cluster...")
     await agent1.shutdown()
     await agent2.shutdown()
     print("✅ Cluster shutdown complete")
