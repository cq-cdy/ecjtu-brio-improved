
from transformers import BartTokenizer, \
    PegasusTokenizer
from modeling_bart import BartScorer
from modeling_pegasus import PegasusScorer

def get_model_tokenizer(model_name,dataset_name,args,is_base_model = False):

    global model,tokenizer
    assert model_name in['bart','pegasus','brio']
    assert dataset_name in ['cnndam', 'xsum']
    if is_base_model:
        assert model_name in['bart','pegasus']
        if model_name == 'bart':
            model = BartScorer.from_pretrained("facebook/bart-base", cache_dir=args.cache_dir)
            tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", cache_dir=args.cache_dir)
        elif model_name =='pegasus':
            model = PegasusScorer.from_pretrained("google/pegasus-x-base", cache_dir=args.cache_dir)
            tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-x-base", cache_dir=args.cache_dir)
        else:
            raise "base model can't matched !"
        return  model,tokenizer
    else:
        ## bart 系列
        if(model_name == 'bart' and dataset_name =='cnndam'):

            model = BartScorer.from_pretrained("facebook/bart-large-cnn", cache_dir=args.cache_dir)
            tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn", cache_dir=args.cache_dir)

        elif model_name =='bart' and dataset_name=='xsum':
            model = BartScorer.from_pretrained("facebook/bart-large-xsum", cache_dir=args.cache_dir)
            tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-xsum", cache_dir=args.cache_dir)

        ## pegasus 系列
        elif model_name == 'pegasus' and dataset_name == 'cnndam':
            model = PegasusScorer.from_pretrained('google/pegasus-cnn_dailymail', cache_dir=args.cache_dir)
            tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-cnn_dailymail', cache_dir=args.cache_dir)

        elif model_name == 'pegasus' and dataset_name == 'xsum':
            model = PegasusScorer.from_pretrained('google/pegasus-xsum', cache_dir=args.cache_dir)
            tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum', cache_dir=args.cache_dir)

        ## brio系列
        elif model_name == 'brio' and dataset_name == 'cnndam':
            model = BartScorer.from_pretrained('Yale-LILY/brio-cnndm-uncased', cache_dir=args.cache_dir)
            tokenizer = BartTokenizer.from_pretrained('Yale-LILY/brio-cnndm-uncased', cache_dir=args.cache_dir)

        elif model_name == 'brio' and dataset_name == 'xsum':
            model = PegasusScorer.from_pretrained('Yale-LILY/brio-xsum-cased', cache_dir=args.cache_dir)
            tokenizer = PegasusTokenizer.from_pretrained('Yale-LILY/brio-xsum-cased', cache_dir=args.cache_dir)
        else:
            raise "model_name and dataset_name can't match!"

        return model,tokenizer


