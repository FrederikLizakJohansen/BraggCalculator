from demo.refine_generated_cif import run


def test_compact_refinement_demo_recovers_assignment_and_writes_cif(tmp_path):
    output = tmp_path / "refined.cif"
    result = run(output)
    best = result.species_assignments.candidates[0]

    assert tuple(site.proposed_species for site in best.sites) == ("Cs", "Cl", "Na")
    assert result.fit_statistics["r_wp"] < 0.01
    assert output.read_text(encoding="utf-8") == result.refined_cif
