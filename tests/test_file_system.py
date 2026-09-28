from qurl.definitions import results_dir


def test_results_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv('QURL_RESULTS_DIR', str(tmp_path / 'custom'))
    assert results_dir() == tmp_path / 'custom'


def test_results_dir_defaults_to_cwd(monkeypatch, tmp_path):
    monkeypatch.delenv('QURL_RESULTS_DIR', raising=False)
    monkeypatch.chdir(tmp_path)
    assert results_dir() == tmp_path / 'results'


def test_results_dir_relative_env_resolves_absolute(monkeypatch, tmp_path):
    monkeypatch.setenv('QURL_RESULTS_DIR', 'rel_dir')
    monkeypatch.chdir(tmp_path)
    result = results_dir()
    assert result.is_absolute()
    assert result == tmp_path / 'rel_dir'
