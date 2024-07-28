import q.cli


def test_prompt_alan_turing():
    prompt = "Alan Turing theorized that computers would one day become"
    r = q.cli.run(
        prompt=prompt,
        n_tokens_to_generate=40,
        model_size="124M",
        models_dir="models",
    )

    assert r.tps > 0
    assert (
        r.output_text
        == " the most powerful machines on the planet.\n\n"
        + "The computer is a machine that can perform complex calculations, "
        + "and it can perform these calculations in a way that is very similar to the human brain.\n"
    )
