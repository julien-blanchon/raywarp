"""`settings.render.mapped`: the one rule a foreign render config becomes RenderSettings by."""

from conftest import recipe

from raywarp.settings.render import mapped


def test_none_fields_keep_the_base_and_overrides_win():
  base = recipe("fast")
  out = mapped(base, {"shadows.enabled": None, "aa.spp": 2}, {"aa.spp": 3})
  assert out.shadows.enabled == base.shadows.enabled and out.aa.spp == 3
  assert mapped(base, {"aa.spp": 2}).aa.spp == 2
