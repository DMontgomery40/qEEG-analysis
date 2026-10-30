"""Direct tools cannot change live portal bytes behind the catalogue's back."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def destinations(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    live = home / 'qEEG-analysis' / 'data' / 'portal_patients'
    live.mkdir(parents=True)
    alias = tmp_path / 'alias'
    alias.symlink_to(live, target_is_directory=True)
    scratch = tmp_path / 'scratch' / 'portal_patients'
    scratch.mkdir(parents=True)
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: home))
    return live, alias, scratch


@pytest.mark.asyncio
async def test_writeup_direct_outputs_guarded_but_publisher_and_scratch_work(destinations, monkeypatch):
    from scripts import generate_patient_facing_writeups as script

    chat = AsyncMock(return_value='\n'.join(script._REQUIRED_PATIENT_FACING_HEADINGS))
    monkeypatch.setattr(script, '_chat_with_retries', chat)
    monkeypatch.setattr(script, 'render_patient_facing_markdown_to_pdf', lambda md, path, **kw: path.write_bytes(b'pdf'))
    for root in destinations:
        kwargs = dict(model_id='test', prompt='', temperature=0, max_tokens=1,
                      label='ZZ_01-01-1900', meta={}, md_path=root/'report.md',
                      pdf_path=root/'report.pdf', meta_path=root/'report.json', no_sync=True)
        if root != destinations[2]:
            with pytest.raises(SystemExit, match='catalogue'):
                await script.generate_writeup(None, **kwargs)
            chat.assert_not_awaited()
            assert not (root/'report.md').exists()
        else:
            await script.generate_writeup(None, **kwargs)
            assert (root/'report.pdf').read_bytes() == b'pdf'
    # Replacing outward-pointing symlinks still changes the live directory.
    for key in ('md_path', 'pdf_path', 'meta_path'):
        target = destinations[2] / ('outside-' + key)
        target.write_text('unchanged')
        link = destinations[0] / ('linked-' + key)
        link.symlink_to(target)
        kwargs[key] = link
    with pytest.raises(SystemExit, match='catalogue'):
        await script.generate_writeup(None, **kwargs)
    assert all(kwargs[key].is_symlink() for key in ('md_path', 'pdf_path', 'meta_path'))
    kwargs['md_path'] = destinations[0]/'registered.md'
    assert await script.generate_writeup(None, **kwargs, publisher=lambda md, meta: 'admitted') == 'admitted'
    assert not kwargs['md_path'].exists()


@pytest.mark.asyncio
async def test_single_agent_refuses_live_output_before_generation(destinations, tmp_path, monkeypatch):
    from scripts import generate_single_agent_patient_report as script

    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'patient_label': 'ZZ_01-01-1900'}))
    report = tmp_path/'report'
    report.mkdir()
    (report/'extracted_enhanced.txt').write_text('source')
    (report/'metadata.json').write_text('{}')
    monkeypatch.setattr(script, 'ensure_data_dirs', lambda: None)
    class ReachedGeneration(Exception):
        pass
    def generate(**kw):
        raise ReachedGeneration
    monkeypatch.setattr(script, '_parse_summary_metrics', generate)
    for root in destinations:
        out = root/'new-patient'
        monkeypatch.setattr(sys, 'argv', ['script', '--manifest', str(manifest), '--combined-report-dir', str(report), '--output-dir', str(out)])
        if root != destinations[2]:
            with pytest.raises(SystemExit, match='catalogue'):
                await script.main()
            assert not out.exists()
        else:
            with pytest.raises(ReachedGeneration):
                await script.main()
            assert out.is_dir()


def test_staging_refuses_live_copy_and_allows_scratch(destinations, tmp_path, monkeypatch):
    from scripts import stage_council_artifacts as script

    monkeypatch.setattr(script, 'init_db', lambda: None)
    monkeypatch.setattr(script, 'get_completed_runs_by_patient_label', lambda: {'ZZ_01-01-1900': [('run', 'report')]})
    monkeypatch.setattr(script, 'sync_patient_to_thrylen', lambda label: False)
    for index, root in enumerate(destinations):
        repo = tmp_path/f'repo-{index}'
        (repo/'data'/'artifacts'/'run'/'stage-4').mkdir(parents=True)
        (repo/'data'/'artifacts'/'run'/'stage-4'/'result.md').write_text('council')
        (repo/'data'/'portal_patients').symlink_to(root, target_is_directory=True)
        (root/'ZZ_01-01-1900').mkdir(exist_ok=True)
        monkeypatch.setattr(script, '__file__', str(repo/'scripts'/'stage_council_artifacts.py'))
        target = root/'ZZ_01-01-1900'/'council'/'run'/'stage-4'/'result.md'
        if root != destinations[2]:
            script.stage_artifacts(dry_run=True)
            with pytest.raises(SystemExit, match='catalogue'):
                script.stage_artifacts()
            assert not target.exists()
        else:
            script.stage_artifacts()
            assert target.read_text() == 'council'


def test_compression_refuses_live_temp_or_replacement(destinations, monkeypatch):
    from scripts import compress_portal_videos as script

    stats = script.VideoStats('video', 100, 1, 8, False)
    args = SimpleNamespace(preset='medium', video_bitrate_kbps=10)
    def encode(cmd):
        Path(cmd[-1]).write_bytes(b'compressed')
    monkeypatch.setattr(script, '_run_command', encode)
    monkeypatch.setattr(script, '_probe_video', lambda path: stats)
    for root in destinations:
        path = root/'video.mp4'
        path.write_bytes(b'original')
        if root != destinations[2]:
            with pytest.raises(SystemExit, match='catalogue'):
                script._compress_video(path, stats, args)
            assert list(root.glob('*.compressing.*')) == []
        else:
            output, _ = script._compress_video(path, stats, args)
            assert output.read_bytes() == b'compressed'
        assert path.read_bytes() == b'original'


def test_registration_update_cannot_overwrite_live_portal(destinations, temp_data_dir, monkeypatch):
    from backend import storage
    from scripts import register_patient_file as script

    with storage.session_scope() as session:
        storage.create_patient(session, label='ZZ_01-01-1900', notes='')
    source = temp_data_dir/'source.mp4'
    source.write_bytes(b'original')
    first = script.register_patient_file(patient_label='ZZ_01-01-1900', src_path=source)
    source.write_bytes(b'replacement')
    for root in destinations:
        target = root/'source.mp4'
        target.write_bytes(b'original')
        monkeypatch.setattr(script, 'patient_file_original_path', lambda *args: target)
        if root != destinations[2]:
            with pytest.raises(SystemExit, match='catalogue'):
                script.register_patient_file(patient_label='ZZ_01-01-1900', src_path=source)
            assert target.read_bytes() == b'original'
            with storage.session_scope() as session:
                assert session.get(storage.PatientFile, first['id']).stored_path == first['stored_path']
        else:
            result = script.register_patient_file(patient_label='ZZ_01-01-1900', src_path=source)
            assert result['id'] == first['id']
            assert target.read_bytes() == b'replacement'
