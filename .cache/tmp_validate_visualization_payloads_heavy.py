from pathlib import Path
from LLMBenchCore import ResultPaths as rp
from visualization_utils import generate_threejs_car_path

model = 'zz_local_file_payload_test'
token = rp.set_current_model(model)
try:
    html = generate_threejs_car_path(
        [[float(i), float((i % 7) - 3), 0.0] for i in range(5000)],
        scenario_name='Payload Test',
        obstacles=[{'x': 20.0, 'y': 1.5, 'width': 2.0, 'length': 4.0, 'label': 'car'}])
    payload_dir = Path(rp.model_artifact_dir(model)) / 'report_payloads'
    payload_files = sorted(payload_dir.glob('*.js')) if payload_dir.exists() else []
    print({
        'html_has_fetch': 'fetch(' in html,
        'html_has_script_src_loader': 'payloadScriptSrc' in html,
        'payload_file_count': len(payload_files),
        'sample_payload_file': str(payload_files[0]) if payload_files else None,
    })
finally:
    rp.reset_current_model(token)
