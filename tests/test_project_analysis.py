from tools.project_analysis import analyze_project, _should_skip_dir, _file_stats


class TestShouldSkipDir:
    def test_skips_venv_variants(self):
        assert _should_skip_dir("venv")
        assert _should_skip_dir("venv_pyqt6")
        assert _should_skip_dir(".venv")

    def test_skips_noise(self):
        assert _should_skip_dir("node_modules")
        assert _should_skip_dir("__pycache__")
        assert _should_skip_dir(".git")

    def test_keeps_source_dirs(self):
        assert not _should_skip_dir("src")
        assert not _should_skip_dir("tools")


class TestAnalyzeProject:
    def test_python_project(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask\nrequests\n")
        (tmp_path / "main.py").write_text("print('hi')\n")
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "views.py").write_text("# views\n")

        out = analyze_project(str(tmp_path))
        assert "Python (pip)" in out
        assert "flask" in out                # framework detected
        assert "main.py" in out              # entry point
        assert "pip install -r requirements.txt" in out
        assert "views.py" in out             # tree

    def test_node_project_with_scripts(self, tmp_path):
        (tmp_path / "package.json").write_text(
            '{"dependencies": {"react": "^18"}, "scripts": {"build": "vite build"}}'
        )
        out = analyze_project(str(tmp_path))
        assert "JavaScript/Node" in out
        assert "react" in out
        assert "npm run build" in out

    def test_dotnet_detected_by_glob(self, tmp_path):
        (tmp_path / "Game.csproj").write_text("<Project/>")
        out = analyze_project(str(tmp_path))
        assert "C#/.NET" in out

    def test_nonexistent_path(self):
        assert analyze_project("/no/such/dir/xyz").startswith("Error")

    def test_stats_excludes_venv(self, tmp_path):
        (tmp_path / "real.py").write_text("x = 1\n")
        venv = tmp_path / "venv_pyqt6" / "Lib"
        venv.mkdir(parents=True)
        for i in range(20):
            (venv / f"junk{i}.py").write_text("# junk\n")
        stats = _file_stats(tmp_path)
        # Only the one real file counts; the 20 venv files are excluded.
        assert stats.startswith("1 files")
