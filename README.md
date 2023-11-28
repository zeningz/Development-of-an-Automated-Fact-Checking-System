```angular2html
Requirements

- python 3.8
- nltk
- [PyTorch](https://pytorch.org/) >=1.10.0
- [Transformers](https://huggingface.co/transformers/) >=4.8.1



As for task1
init model
nohup python -u task1/run.py >train.out 2>&1 &

update and retrain model
nohup python -u task1/run.py --model_pt 23-04-21 >train.out 2>&1 &

To predict
nohup python -u task1/predict.py -p --model_pt 23-04-21 >test_dpr.out 2>&1 &





As for task2
init model
nohup python -u task2/run.py >train.out 2>&1 &

update and retrain model
nohup python -u task2/run.py --model_pt 23-04-21 >train.out 2>&1 &

To predict
nohup python -u task2/run.py -p --model_pt cls >test.out 2>&1 &
```



