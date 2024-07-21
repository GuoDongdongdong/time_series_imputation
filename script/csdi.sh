cd ..
lrs="1e-6"
for lr in $lrs;
do
python main.py -train True -model CSDI -seq_len 48 -lr $lr
done