@mkdir scp_mlf
set c1="G:\rsync\RESEARCHS\table_detection\source_code\github\x270eclipse22_ws\test_01\run_labelme_parser"
set c2=-j ..\..\data\labelme\answer_sheet\IMG-20240101-WA0113.json
set c3=-od scp_mlf
set c4=-smd ./tmp

set c5=-is ..\..\data\smaller\answer_sheet\24-01\280_IMG-20240101-WA0113.jpeg

%c1% %c2% %c3% %c4% %c5%

