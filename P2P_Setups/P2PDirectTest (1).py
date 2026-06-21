import asyncio
import time
import AbstractIntegratedModule
from AbstractIntegratedModule import (
    IntegratedPipeline,
    PipelinePredictionManager,
    CohesiveAgentDeployment
)

SECRET_KEY = "test-secret-key"

LABEL_FILE = "ManualsTraining.txt"
PEER_FILE = "peer_config.json"

memory_name = "arm64_test"

pipeline = IntegratedPipeline(memory_name=memory_name, use_async=True, agent_port=8080)

predict_manager = PipelinePredictionManager(
    pipeline,
    label_csv=LABEL_FILE,
    target_title="window_title",
    label="label"
)

titles, y, label_map = predict_manager.load_labels_from_csv(
    LABEL_FILE,
    "window_title",
    "label"
)

pipeline.train(titles, y)

test_titles = [
    ("Programming in VSCode", "focused_work"),
    ("Watching YouTube", "entertainment")
]

print('FILE:', AbstractIntegratedModule.__file__)

async def main(main_pipeline):
    sec_pipeline = IntegratedPipeline(memory_name=memory_name, use_async=True, agent_port=8010)

    rules = [
        # === WORK / PRODUCTIVITY ===
        (r'code|programming|develop|debug|compile|script', 'focused_work'),
        (r'vscode|visual_studio|ide|terminal|shell', 'focused_work'),
        (r'notion|evernote|onenote|notes|todo|task', 'productive'),
        (r'slack|teams|discord|zoom|meeting|call', 'communication'),
        (r'email|gmail|outlook|inbox|mail', 'communication'),
        
        # === ENTERTAINMENT ===
        (r'youtube|netflix|twitch|stream|video', 'entertainment'),
        (r'music|spotify|soundcloud|audio|player', 'entertainment'),
        (r'game|gaming|steam|epic|play', 'gaming'),
        (r'facebook|instagram|tiktok|social|post', 'social_media'),
        
        # === BROWSING ===
        (r'chrome|firefox|edge|safari|browser', 'browsing'),
        (r'google|search|wiki|wiki|article', 'information'),
        (r'stackoverflow|github|docs|documentation', 'research'),
        
        # === FILE MANAGEMENT ===
        (r'download|folder|file|document|pdf', 'file_work'),
        (r'dropbox|onedrive|google_drive|cloud', 'cloud_storage'),
        (r'zip|rar|extract|compress|archive', 'file_management'),
        
        # === SYSTEM / DEV ===
        (r'terminal|cmd|powershell|bash|shell', 'system_work'),
        (r'docker|kubernetes|container|deploy', 'devops'),
        (r'git|commit|push|pull|branch|merge', 'version_control'),
        (r'test|unit|debug|error|exception', 'testing'),
        
        # === DATA / ANALYSIS ===
        (r'excel|spreadsheet|sheet|csv|table', 'data_work'),
        (r'python|r|sql|query|database', 'data_analysis'),
        (r'chart|graph|visualization|dashboard|plot', 'visualization'),
        
        # === COMMUNICATION ===
        (r'whatsapp|telegram|signal|messenger', 'messaging'),
        (r'zoom|meet|webex|video_call', 'video_call'),
        (r'calendar|schedule|event|meeting|appointment', 'scheduling'),
        
        # === CREATIVE ===
        (r'photoshop|illustrator|figma|design|canvas', 'creative'),
        (r'premiere|final_cut|video_edit|render', 'video_editing'),
        (r'blender|3d|model|render|animation', '3d_work'),
        
        # === LEARNING ===
        (r'coursera|udemy|edx|course|learn', 'learning'),
        (r'book|ebook|reader|pdf|document', 'reading'),
        (r'podcast|audiobook|listen|lecture', 'audio_learning'),
        
        # === UTILITY ===
        (r'calculator|converter|tool|utility', 'utility'),
        (r'weather|clock|timer|alarm|reminder', 'utility'),
        (r'translate|language|dictionary|translate', 'utility'),
        
        # === RARITY PATTERNS ===
        (r'common|not_common|twitch|debian|watch', 'very abundant'),
        (r'bit-common|pycharm|unix|code|programming|python|java', 'bit-abundant'),
        (r'medium|discord|teams|zoom|linux_mint|message', 'abundant'),
        (r'rare|pdf|word|macOS|ubuntu|document', 'not abundant'),
        (r'ultra|firefox|edge|browser|unix|web', 'medium rare'),
        (r'ultra_rare|music|linux|Home_linux_router', 'bit-rare'),
        (r'medium-rare|steam|red_hat_enterprise_linux|play|windows', 'very rare'),
        (r'rarer|oracle|system|config|server_linux_router', 'absolute rare'),
    ]

    agent1 = CohesiveAgentDeployment(
        pipeline=main_pipeline,
        memory_name="agent1",
        filename=LABEL_FILE,
        target_title="window_title",
        label_name="label",
        enable_peers=True,
        peer_discovery_port=5555,
        secret_key=SECRET_KEY,
        shared_auth_token=SECRET_KEY,
        predict_manager=predict_manager,
        peer_config=PEER_FILE,
    )

    agent2 = CohesiveAgentDeployment(
        pipeline=sec_pipeline,
        memory_name="agent2",
        filename=LABEL_FILE,
        target_title="window_title",
        label_name="label",
        enable_peers=True,
        peer_discovery_port=5556,
        secret_key=SECRET_KEY,
        shared_auth_token=SECRET_KEY,
        predict_manager=predict_manager,
        peer_config=PEER_FILE,
    )

    await agent1.start()
    print("Agent 1 started on port 5555, Time:", time.time())
    await agent2.start()
    print("Agent 2 started on port 5556, Time:", time.time())

    await asyncio.sleep(3)

    api_key = agent1.get_api_key()

    texts = {
        "test_titles": test_titles,
        "label_map": label_map,
        "rules": rules,
        "X": None,
        "y": None, # X and y are optional samples.
        "use_transformer": False,
        "agent_id": "arm64-demo"
    }

    agent1.pipeline.use_transformer = False
    agent2.pipeline.use_transformer = False

    result = await agent1.multi_modal_peer_ensemble_prediction(
        texts=texts,
        api_key=api_key,
        method="advanced",
        disable_sync=True
    )

    result2 = await agent2.multi_modal_peer_ensemble_prediction(
        texts=texts,
        api_key=api_key,
        method="advanced",
        disable_sync=True
    )    

    print("\n=== RESULT ===")
    print(f"\n📊 Ensemble Result for Agent 1:")
    print(f"   Prediction: {result.get('prediction', 'N/A')}")
    print(f"   Confidence: {result.get('confidence', 0):.2%}")

    print('[=] For agent 2: ')
    print(f"   Second Prediction: {result2.get('prediction', 'N/A')}")
    print(f"   Second Confidence: {result2.get('confidence', 0):.2%}")

    agent1._peer_agent.stop_server()
    agent2._peer_agent.stop_server()

    await agent1.shutdown()
    await agent2.shutdown()

asyncio.run(main(pipeline))
