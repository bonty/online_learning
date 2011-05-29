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
my $bias = 0.0;
my $method = 'P';

GetOptions("train" => \$train,
           "test" => \$test,
           "eta=f" => \$eta,
           "iter=i" => \$iter,
           "bias=f" => \$bias,
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

    $fvalue{0} = $bias;
    foreach my $val (@features) {
      my ($id, $v) = split /:/, $val;
      $fvalue{$id+1} = $v;
      $max_fid = $id if ($id > $max_fid);
    }

    $expl[$size][0] = $label eq "+1" ? 1 : -1;
    $expl[$size][1] = \%fvalue;
    $size++;
  }
  close $dh;

  # initialize weight
  my @w = (0) x ($max_fid + 2);
  my @tmp = (0) x ($max_fid + 2);
  my $cnt = 0;

  for my $i (1 .. $iter) {
    # suffle training data
    shuffleArray(\@expl);

    my $error = 0;
    for my $i (0 .. $#expl) {
      my $l =  $expl[$i][0];
      my $fs = $expl[$i][1];
      my $score = 0;

      foreach my $f (keys %$fs) {
        $score += $$fs{$f} * $w[$f];
      }

      foreach my $f (keys %$fs) {
        if ($method eq 'P') {
          $w[$f] += $eta*$l*$$fs{$f} if ($l * $score <= 0);
        }
        elsif ($method eq 'AP1') {
          if ($l * $score <= 0) {
            $w[$f] += $eta*$l*$$fs{$f};
          }
          else {
            $tmp[$f] += $w[$f];
          }
        }
        elsif ($method eq 'AP2') {
          $w[$f] += $eta*$l*$$fs{$f} if ($l * $score <= 0);
          $tmp[$f] += $w[$f];
        }
      }
      $cnt++ if ($l * $score > 0 && $method eq 'AP1');
      $cnt++ if ($method eq 'AP2');
    }
  }

  if ($method eq 'AP1' || $method eq 'AP2') {
    for my $f (0 .. $#w) {
      $w[$f] = $tmp[$f] / $cnt;
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
      $score += $w[$id] * $v if ($w[$id]);
    }

    my $estimate = $score >=0 ? "+1" : "-1";

    # print "j:$l\te:$estimate $score\n";

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
