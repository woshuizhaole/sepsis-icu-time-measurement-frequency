from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('stage5_wbm_impl','/data/workplace/wyx/measurement_intensity_ssl_20261015/code/analysis/13_stage5_wbm.py'); m=module_from_spec(spec); spec.loader.exec_module(m)
WBMGRU=m.WBMGRU; WinDS=m.WinDS; make_groups=m.make_groups; BASE=m.BASE; INPUT=m.INPUT; DYN=m.DYN; MASK_SL=m.MASK_SL

