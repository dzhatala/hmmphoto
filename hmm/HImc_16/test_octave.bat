set octave=C:\rps\Octave-9.4.0\mingw64\bin\octave.bat
@rem call C:\rps\Octave-9.4.0\octave G:\rsync\RESEARCHS\table_detection\source_code\github\tablerec\octave\gen_obs_multi.m tmp\x.jpeg tmp\
@rem set scr_m=G:\rsync\RESEARCHS\table_detection\source_code\github\tablerec\octave\gen_obs_multi.m
set scr_m=gen_obs_multi.m
@rem set scr_m=im_raw_01.m
dir %scr_m%
@rem 
call %octave% %scr_m% .\tmp\x.jpeg .\tmp
@rem call %octave% im_raw_01.m
