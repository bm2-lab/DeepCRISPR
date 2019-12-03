# Example Description

## File list
File name | Task | Task type
--------|--------|---------
eg_cls_on_target.episgt | On-target | Classification
eg_reg_on_target.repisgt | On-target | Regression
eg_reg_on_target_seq.rsgt | On-target | Seq-Only Regression
eg_cls_off_target.epiotrt | Off-target | Classification
eg_reg_off_target.repiotrt | Off-target | Regression


## On-target fields

Chrom|Start|End|Strand|Target Seq|CTCF|Dnase|H3K4me3|RRBS|Label
---|---|---|---|---|---|---|---|---|---
chr17|33469132|33469154|-|CTTGCTCGCGCAGGACGAGGCGG|AAAAAAAAAAAAAAAAAAAAAAA|AAAAAAAAAAAAAAAAAAAAAAA|AAAAAAAAAAAAAAAAAAAAAAA|NNNNNNNNNNNNNNNNNNNNNNN|1

## On-target Seq-Only fields

Chrom|Start|End|Strand|Target Seq|Label
---|---|---|---|---|---
chr17|33469132|33469154|-|CTTGCTCGCGCAGGACGAGGCGG|1


## Off-target fields

Id|Target Seq|Target CTCF|Target Dnase|Target H3K4me3|Target RRBS|Off-target Seq|Off-target CTCF|Off-target Dnase|Off-target H3K4me3|Off-target RRBS| Label
---|---|---|---|---|---|---|---|---|---|---|---
sg9|AAATGAGAAGAAGAGGCACAGGG|AAAAAAAAAAAAAAAAAAAAAAA|AAAAAAAAAAAAAAAAAAAAAAA|AAAAAAAAAAAAAAAAAAAAAAA|NNNNNNNNNNNNNNNNNNNNNNN|GCATGAGAAGAAGAGACATAGCC|AAAAAAAAAAAAAAAAAAAAAAA|AAAAAAAAAAAAAAAAAAAAAAA|AAAAAAAAAAAAAAAAAAAAAAA|NNNNNNNNNNNNNNNNNNNNNNN|0

Note:
For epigenetic features including CTCF, Dnase, H3K4me3 and RRBS, "A" stands for 1 representing signal located and "N" stands for 0 representing no signal.
