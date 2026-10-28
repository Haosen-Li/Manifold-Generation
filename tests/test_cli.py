import json

from manifold_gen.cli import main, read_prompts


def test_prompt_formats(tmp_path):
    for name, text in {
        "p.txt": "a\nb\n",
        "p.json": '["a", "b"]',
        "p.jsonl": '{"caption":"a"}\n{"caption":"b"}\n',
        "p.csv": "caption\na\nb\n",
    }.items():
        path = tmp_path / name
        path.write_text(text)
        assert read_prompts(path) == ["a", "b"]


def test_model_listing(capsys):
    main(["--list-models"])
    output = capsys.readouterr().out
    assert "hunyuan-dit" in output and "modelscope" in output


def test_cli_writes_real_image_and_metadata(monkeypatch, tmp_path, tokenizer):
    from conftest import make_pipeline

    import manifold_gen.cli as cli
    from manifold_gen import plug

    pipe, _ = make_pipeline("sdxl", tokenizer)

    def load(*args, **kwargs):
        return plug(pipe, samg=kwargs["samg"], oracle_noise=kwargs["oracle_noise"], z2=kwargs["z2"])

    monkeypatch.setattr(cli, "load_pipeline", load)
    main(
        [
            "--model",
            "sdxl",
            "--prompt",
            "a b",
            "--components",
            "samg",
            "--height",
            "16",
            "--width",
            "16",
            "--steps",
            "2",
            "--device",
            "cpu",
            "--output",
            str(tmp_path),
            "--compare-baseline",
        ]
    )
    assert (tmp_path / "00000_result.png").is_file()
    assert (tmp_path / "00000_baseline.png").is_file()
    metadata = json.loads((tmp_path / "run.json").read_text())
    assert len(metadata["records"]) == 2
    assert metadata["records"][1]["stats"]["denoiser_calls"] == 2
