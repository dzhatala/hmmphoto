<?php
function print_help(){
	echo "php create_initial_proto_hmm.php { -h | --help }\n";
	echo "php create_initial_proto_hmm.php  vector_size [ num_state] }";
	
}


function create_transp5(){
	echo "<TransP> 5\n";
	echo "0.0 1.0 0.0 0.0 0.0\n";
	echo "0.0 0.6 0.4 0.0 0.0\n";
	echo "0.0 0.0 0.6 0.4 0.0\n";
	echo "0.0 0.0 0.0 0.7 0.3\n";
	echo "0.0 0.0 0.0 0.0 0.0\n";
	echo "<EndHMM>\n";

}

function create_transp20(){
	echo "<TransP> 5\n";
	echo "0.0 1.0 0.0 0.0 0.0\n";
	echo "0.0 0.6 0.4 0.0 0.0\n";
	echo "0.0 0.0 0.6 0.4 0.0\n";
	echo "0.0 0.0 0.0 0.7 0.3\n";
	echo "0.0 0.0 0.0 0.0 0.0\n";
	echo "<EndHMM>\n";

}




 //echo $argv[1]; return 
// echo $argc; return ;
if ( $argc<2){
	print_help();
	return ;
}
$vs=intval($argv[1]);  
 if ($vs<=0 ) {
	 print_help(); return ;
 }
	 
	 // echo $vs ; return ;
?>
<?php
	$numstates=5;
	$hmm="\"php_init_19\"";
echo "~o <VecSize>  $vs <MFCC>\n";
echo "~h ".$hmm."\n";
echo "<BeginHMM>\n<NumStates> ".$numstates."\n";
for ($istate=2; $istate<$numstates ;$istate++){

	echo "<State> ".$istate."\n";
	echo "<Mean> ". $vs ."\n"; 
	for ($x=1;$x<=$vs;$x++) echo "0.0 " ;
	echo "\n";
	echo "<Variance> ".$vs."\n" ;
	for ($x=1;$x<=$vs;$x++) echo "1.0 " ;
	echo "\n";
}
?>
<TransP> 5
0.0 1.0 0.0 0.0 0.0
0.0 0.6 0.4 0.0 0.0
0.0 0.0 0.6 0.4 0.0
0.0 0.0 0.0 0.7 0.3
0.0 0.0 0.0 0.0 0.0
<EndHMM>
