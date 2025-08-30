<?php //echo $argv[1]; return 
?>
<?php $vs=$argv[1]; //TODO NOT used yet ?>
~o <VecSize> <?php echo $vs ?> <MFCC>
~h "initial_0_D_A"
<BeginHMM>
<NumStates> 5
<State> 2
<Mean> <?php echo $vs ?> 
<?php
for ($x=1;$x<=$vs;$x++) echo "0.0 " ;
?> 
<Variance> <?php echo $vs."\n" ?>
<?php
for ($x=1;$x<=$vs;$x++) echo "1.0 " ;
?> 
<State> 3
<Mean> <?php echo $vs."\n" ?>
<?php
for ($x=1;$x<=$vs;$x++) echo "0.0 " ;
?>
<Variance> <?php echo $vs."\n" ?> 
<?php
for ($x=1;$x<=$vs;$x++) echo "1.0 " ;
?>
<State> 4 
<Mean> <?php echo $vs."\n" ?>
<?php
for ($x=1;$x<=$vs;$x++) echo "0.0 " ;
?>
<Variance> <?php echo $vs."\n" ?> 
<?php
for ($x=1;$x<=$vs;$x++) echo "1.0 " ;
?>
<TransP> 5
0.0 1.0 0.0 0.0 0.0
0.0 0.6 0.4 0.0 0.0
0.0 0.0 0.6 0.4 0.0
0.0 0.0 0.0 0.7 0.3
0.0 0.0 0.0 0.0 0.0
<EndHMM>
