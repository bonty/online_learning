#!/usr/local/bin/perl
#
# perceptron
#
# usage:
# training : perl perceptron.pl -train <trainingdata> <modeloutput>
# test     : perl perceptron.pl -test <model> <testdata>

use strict;
use warnings;
use Getopt::Long;

sub usage {
  print <<EOS;
Usage:
  training : perl perceptron.pl -train <trainingdata> <modeloutput>
  test     : perl perceptron.pl -test <model> <testdata>
EOS
  exit 1;
}

my $train;
my $test;

# params
my $eta = 1.0;
my $iter = 20;
my $method = 'simple';

GetOptions("train" => \$train,
           "test" => \$test,
           "eta=f" => \$eta,
           "iter=i" => \$iter,
           "method=s" => \$method);

if (!$train && !$test) {
  usage();
}

if ($train) {
  if (@ARGV != 2) {
    usage();
  }
  my $data = shift or die "$!\n";
  my $model = shift or die "$!\n";

 my @expl = (); # example
 my $size = 0;  # number of example
 my $max_fid = 0; # max value of feature id

  open my $dh, '<', $data or die "$data: $!\n";

  while (<$dh>) {
    chomp;
    my $score = 0;
    my ($label, @features) = split / /;
    my %fvalue = ();

    # $fvalue{0} = 1;
    foreach my $val (@features) {
      my ($id, $v) = split /:/, $val;
      $fvalue{$id} = $v;
      $max_fid = $id if ($id > $max_fid);
    }

    $expl[$size][0] = $label eq "+1" ? 1 : -1;
    $expl[$size][1] = \%fvalue;
    $size++;
  }
  close $dh;

  # initialize weight
  my @w = (0) x ($max_fid + 1);
  my @tmp = (0) x ($max_fid + 1);
  my $cnt = 0;

  for my $i (1 .. $iter) {
    # suffle training data
    shuffleArray(\@expl);

    foreach my $val (@expl) {
      my $l =  $$val[0];
      my $fs = $$val[1];
      my $score = 0;

      foreach my $f (keys %$fs) {
        $score = $$fs{$f} * $w[$f];
      }

      foreach my $f (keys %$fs) {
        $w[$f] += $eta*$l*$$fs{$f} if ($l * $score <= 0);
      }
    }
  }

  open my $mo, '>', $model or die "$!\n";
  for my $i (0 .. $#w) {
    print {$mo} "$w[$i]\n";
  }
  close $mo;
}
elsif ($test) {
  if (@ARGV != 2) {
    usage();
  }
  my $model = shift or die "$!\n";
  my $data = shift or die "$!\n";

  my @w = ();

  open my $mo, '<', $model or die "$model: $!\n";
  while (<$mo>) {
    chomp;
    push @w, $_;
  }
  close $mo;

  my $num = 0;
  my $curr = 0;

  open my $dh, '<', $data or die "$model: $!\n";
  while (<$dh>) {
    chomp;
    my ($l, @fs) = split / /;
    my $score = 0;

    foreach my $val (@fs) {
      my ($id, $v) = split /:/, $val;
      $score = $w[$id] * $v if ($w[$id]);
    }

    my $estimate = $score >=0 ? "+1" : "-1";

    $num++;
    $curr++ if ($l eq $estimate);
  }
  close $dh;

  #printf("precision: %.3f\n", $curr / $num);
  printf("%.3f\n", $curr / $num);
}

sub shuffleArray {
  my $array = shift;
  for (my $i = @$array; --$i;) {
    my $j = int rand($i + 1);
    next if ($i == $j);
    @$array[$i, $j] = @$array[$j, $i];
  }
}
