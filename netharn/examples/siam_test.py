import os
import ubelt as ub
import numpy as np
import netharn as nh
import torch
import torchvision
import itertools as it
import utool as ut
import glob
from collections import OrderedDict
import parse
def _auto_argparse(func):
    """
    Transform a function with a Google Style Docstring into an
    `argparse.ArgumentParser`.  Custom utility. Not sure where it goes yet.
    """
    from xdoctest import docscrape_google as scrape
    import argparse
    import inspect

    # Parse default values from the function dynamically
    spec = inspect.getargspec(func)
    kwdefaults = dict(zip(spec.args[-len(spec.defaults):], spec.defaults))

    # Parse help and description information from a google-style docstring
    docstr = func.__doc__
    description = scrape.split_google_docblocks(docstr)[0][1][0].strip()
    google_args = {argdict['name']: argdict
                   for argdict in scrape.parse_google_args(docstr)}

    # Create the argument parser and register each argument
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for arg in spec.args:
        argkw = {}
        if arg in kwdefaults:
            argkw['default'] = kwdefaults[arg]
        if arg in google_args:
            garg = google_args[arg]
            argkw['help'] = garg['desc']
            try:
                argkw['type'] = eval(garg['type'], {})
            except Exception:
                pass
        parser.add_argument('--' + arg, **argkw)
    return parser


def fit(dbname='PZ_MTEST', nice='untitled', dim=416, bsize=6, bstep=4,
        lr=0.001, decay=0.0005, workers=0, xpu='cpu', epoch='best', thres=0.5):
    """
    Train a siamese chip descriptor for animal identification.

    Args:
        dbname (str): Name of IBEIS database to use
        nice (str): Custom tag for this run
        dim (int): Width and height of the network input
        bsize (int): Base batch size. Number of examples in GPU at any time.
        bstep (int): Multiply by bsize to simulate a larger batches.
        lr (float): Base learning rate
        decay (float): Weight decay (L2 regularization)
        workers (int): Number of parallel data loader workers
        xpu (str): Device to train on. Can be either `'cpu'`, `'gpu'`, a number
            indicating a GPU (e.g. `0`), or a list of numbers (e.g. `[0,1,2]`)
            indicating multiple GPUs
      #  epoch (int): epoch number to evaluate on
      #  thres (float): threshold for accuracy and mcc calculation
    """

    # There has to be a good way to use argparse and specify params only once.
    # Pass all args down to comparable_vamp
    import inspect
    kw = ub.dict_subset(locals(), inspect.getargspec(fit).args)
    comparable_vamp(**kw)

def comparable_vamp(**kwargs):

    import parse
    import glob
    from ibeis.algo.verif import vsone
    parse.log.setLevel(30)
    from netharn.examples.siam_ibeis import randomized_ibeis_dset
    from netharn.examples.siam_ibeis import SiameseLP, SiamHarness, setup_harness
    dbname = ub.argval('--db', default='GZ_Master1')
    nice = ub.argval('--nice',default='untitled')
   # thres = ub.argval('--thres',default=0.5)
    
    dim = 512
    datasets = randomized_ibeis_dset(dbname, dim=dim)

    class_names = ['diff', 'same']
    workdir = ub.ensuredir(os.path.expanduser(
        '~/data/work/siam-ibeis2/' + dbname))
    task_name = 'binary_match'

    datasets['test'].pccs
    datasets['train'].pccs

    # pblm = vsone.OneVsOneProblem.from_empty('PZ_MTEST')
    ibs = datasets['train'].infr.ibs
    labeled_aid_pairs = [datasets['train'].get_aidpair(i)
                         for i in range(len(datasets['train']))]
    pblm_train = vsone.OneVsOneProblem.from_labeled_aidpairs(
        ibs, labeled_aid_pairs, class_names=class_names,
        task_name=task_name,
    )

    test_labeled_aid_pairs = [datasets['test'].get_aidpair(i)
                              for i in range(len(datasets['test']))]
    pblm_test = vsone.OneVsOneProblem.from_labeled_aidpairs(
        ibs, test_labeled_aid_pairs, class_names=class_names,
        task_name=task_name,
    )


    harn = setup_harness(dbname=dbname)
    harn.initialize()

    margin = harn.hyper.criterion_params['margin']
  
    vamp_res = vamp(pblm_train, workdir, pblm_test)
  
 # ----------------------------
    # Evaluate the siamese dataset
    pretrained = 'resnet50'
    branch = getattr(torchvision.models, pretrained)(pretrained=False)
    model = SiameseLP(p=2, branch=branch, input_shape=(1, 3, dim, dim))
    #if torch.cuda.is_available():
    xpu = nh.XPU.cast(kwargs.get('xpu','cpu'))#xpu_device.XPU.from_argv()
    print('Preparing to predict {} on {}'.format(model.__class__.__name__,xpu))

    xpu.move(model)
    train_dpath ='/home/angelasu/work/siam-ibeis2/' + dbname + '/fit/nice/' + nice
    print(train_dpath)
    epoch = ub.argval('--epoch', default=None)
    epoch = int(epoch) if epoch is not None and epoch != 'best' and epoch != 'recent' and epoch != 'all'  else epoch
    max_roc = 0
    siam_res_arr = []
    dist_arr_ret = []
    if epoch == 'all':
       # max_roc = 0
       # siam_res_arr = []
        for  file in sorted(glob.glob(train_dpath + '/*/_epoch_*.pt')):
            print(file)
            load_path = file
            dist_arr, max_roc, siam_res_arr = siam(load_path, xpu, model, pblm_test, datasets, margin, max_roc, siam_res_arr, dist_arr_ret)
        siam_res = siam_res_arr[-1]
    else:
        load_path = get_snapshot(train_dpath, epoch=epoch)
        dist_arr, siam_res = siam(load_path, xpu, model, pblm_test, datasets, margin, max_roc, siam_res_arr, dist_arr_ret)
    thres = ub.argval('--thres', default=0.5)
    thres = float(thres)
    thres_range = np.linspace(thres-0.05, thres+0.05,41)
    for val in thres_range:
        print('threshold value = {!r}'.format(val))
        p_same = torch.sigmoid(torch.Tensor(-(dist_arr-margin))).numpy()-(val-0.5)
        p_diff = 1 - p_same
   # y_pred = (dist_arr <= 4)

        import pandas as pd
        pd.set_option("display.max_rows", None)
    # hack probabilities
        probs_df = pd.DataFrame(
       # np.array([dist_arr,p_same]).T,
        np.array([p_diff,p_same]).T,
       # np.array([y_pred,y_pred]).T,
        columns=class_names,
        index=pblm_test.samples['binary_match'].indicator_df.index
    )

        sorted_df = probs_df.sort_index()
       # sorted_df = probs_df.sort_values(by='same', ascending=False)
        sorted_df = sorted_df.iloc[:,1:2] #0:2
   # sorted_df.info()

        siam_res = vsone.clf_helpers.ClfResult()
        siam_res.probs_df = probs_df
        siam_res.probhats_df = None
        siam_res.data_key = 'SiamL2'
        siam_res.feat_dims = None
        siam_res.class_names = class_names
        siam_res.task_name = task_name
        siam_res.target_bin_df = pblm_test.samples['binary_match'].indicator_df
        siam_res.target_enc_df = pblm_test.samples['binary_match'].encoded_df

        target_sort = siam_res.target_bin_df.sort_index().iloc[:,1:2]
        result = pd.concat([sorted_df, target_sort],axis=1)
        siam_report = siam_res.extended_clf_report() 
        print('siam roc = {}'.format(siam_res.roc_score())) 
    print(nice)
    print('--- SIAM ---')
    print('epoch = {!r}'.format(epoch))
    print('margin= {!r}'.format(margin))
    print('threshold = {!r}'.format(thres))
    siam_report = siam_res.extended_clf_report()  # NOQA
    print('siam roc = {}'.format(siam_res.roc_score()))
   # siam_res.show_roc('same')
   # ut.show_if_requested()
   # siam_res.show_roc('diff')
   # ut.show_if_requested()

   #from sklearn import metrics
   #/ print('mcc {}'.format(metrics.matthews_corrcoef(siam_res.target_bin_df, probs_df)))
    print('--- VAMP ---')
    vamp_report = vamp_res.extended_clf_report()  # NOQA
    print('vamp roc = {}'.format(vamp_res.roc_score()))

def get_snapshot(train_dpath, epoch='recent'):
    """
    Get a path to a particular epoch or the most recent one
    """
    snapshots = sorted(glob.glob(train_dpath + '/*/_epoch_*.pt'))
    if epoch is None:
        epoch = 'recent'
    if epoch == 'best':
        snapshots = sorted(glob.glob(train_dpath + '/best_snapshot.pt'))
        load_path = snapshots[0]
    elif epoch == 'recent':
        load_path = snapshots[-1]
    else:
        snapshot_nums = [parse.parse('{}_epoch_{num:d}.pt', path).named['num']
                         for path in snapshots]
        load_path = dict(zip(snapshot_nums, snapshots))[epoch]
    print('load path: ',load_path) 
    return load_path

def siam(load_path, xpu, model, pblm_test, datasets,margin, max_roc, siam_res_arr, dist_arr_ret):
    from netharn.examples.siam_ibeis import randomized_ibeis_dset
    from netharn.examples.siam_ibeis import SiameseLP, SiamHarness, setup_harness
    from ibeis.algo.verif import vsone
    parse.log.setLevel(30)
    dbname = ub.argval('--db', default='GZ_Master1')
    nice = ub.argval('--nice',default='untitled')
   # thres = ub.argval('--thres',default=0.5)

    dim = 512

    class_names = ['diff', 'same']
    workdir = ub.ensuredir(os.path.expanduser(
        '~/data/work/siam-ibeis2/' + dbname))
    task_name = 'binary_match'

    # pblm = vsone.OneVsOneProblem.from_empty('PZ_MTEST')
   # print('Preparing to predict {} on {}'.format(model.__class__.__name__,xpu))

    xpu.move(model)

    # ----------------------------
    # Evaluate the siamese dataset
    thres = ub.argval('--thres', default=0.5)
    thres = float(thres)
   # print('Loading snapshot onto {}'.format(xpu))
    'pretrained model'
    snapshot = torch.load(load_path, map_location= lambda storage, loc:storage)
    #map_location = xpu.map_location())
    #map_location={'cuda:1': 'cpu'})

    new_pretrained_state = OrderedDict()
    for k, v in snapshot['model_state_dict'].items():
        layer_name = k.replace("module.", "")
        new_pretrained_state[layer_name] = v

    model.load_state_dict(new_pretrained_state)

    del snapshot

    model.train(False)

    dists = []
    dataset = datasets['test']
    
    #for aid1, aid2 in ub.ProgIter(pblm_train.samples.index, label='training set'):
#        print(aid1, aid2)

   # for aid1, aid2 in ub.ProgIter(pblm_test.samples.index, label='predicting'):
    for aid1, aid2 in ub.ProgIter(pblm_test.samples.index, label='predicting'): 
        img1, img2 = dataset.load_from_edge(aid1, aid2)
        img1 = torch.FloatTensor(img1.transpose(2, 0, 1))
        img2 = torch.FloatTensor(img2.transpose(2, 0, 1))
        #img1, img2 = xpu.variables(*inputs)
        img1 = xpu.variable(img1)
        img2 = xpu.variable(img2)
        dist_tensor = model(img1[None, :], img2[None, :])
        dist = dist_tensor.data.cpu().numpy()
        dists.append(dist)

    dist_arr= np.squeeze(np.array(dists))
    
    #p_same = np.exp(-dist_arr)
    #p_diff = 1 - p_same

    thres = ub.argval('--thres', default=0.5)
    thres = float(thres)
    p_same = torch.sigmoid(torch.Tensor(-(dist_arr-margin))).numpy()-(thres-0.5)
    p_diff = 1 - p_same
   # y_pred = (dist_arr <= 4)

    import pandas as pd
    pd.set_option("display.max_rows", None)
    # hack probabilities
    probs_df = pd.DataFrame(
       # np.array([dist_arr,p_same]).T,
        np.array([p_diff,p_same]).T,
       # np.array([y_pred,y_pred]).T,
        columns=class_names,
        index=pblm_test.samples['binary_match'].indicator_df.index
    )

    sorted_df = probs_df.sort_index()
  #  sorted_df = probs_df.sort_values(by='same', ascending=False)
    sorted_df = sorted_df.iloc[:,1:2] #0:2
    #print(sorted_df)
    # sorted_df.info()

    siam_res = vsone.clf_helpers.ClfResult()
    siam_res.probs_df = probs_df
    siam_res.probhats_df = None
    siam_res.data_key = 'SiamL2'
    siam_res.feat_dims = None
    siam_res.class_names = class_names
    siam_res.task_name = task_name
    siam_res.target_bin_df = pblm_test.samples['binary_match'].indicator_df
    siam_res.target_enc_df = pblm_test.samples['binary_match'].encoded_df

    target_sort = siam_res.target_bin_df.sort_index().iloc[:,1:2]
    result = pd.concat([sorted_df, target_sort],axis=1,sort=True)
    #print(result) 
    epoch = ub.argval('--epoch',default=None)
    if epoch == 'all':
        print('siam roc = {}'.format(siam_res.roc_score()))
        if siam_res.roc_score() > max_roc:
            max_roc = siam_res.roc_score()
           # siam_res_arr = []
            siam_res_arr.append(siam_res)
            dist_arr_ret.append(dist_arr)
        return dist_arr, max_roc, siam_res_arr
    else:
        return dist_arr, siam_res

def vamp(pblm_train, workdir, pblm_test):
    # ----------------------------
    # Build a VAMP classifier using the siamese training dataset
    pblm_train.load_features()
    pblm_train.samples.print_info()
    pblm_train.build_feature_subsets()
    pblm_train.samples.print_featinfo()

    pblm_train.learn_deploy_classifiers(task_keys=['binary_match'])
    clf_dpath = ub.ensuredir((workdir, 'clf'))
    classifiers = pblm_train.ensure_deploy_classifiers(dpath=clf_dpath)
    ibs_clf = classifiers['binary_match']
    clf = ibs_clf.clf

    # ----------------------------
    # Evaluate the VAMP classifier on the siamese testing dataset
    pblm_test.load_features()
    pblm_test.samples.print_info()
    pblm_test.build_feature_subsets()
    pblm_test.samples.print_featinfo()
    data_key = pblm_train.default_data_key
    task_key = 'binary_match'
    vamp_res = pblm_test._external_classifier_result(clf, task_key, data_key)
    vamp_report = vamp_res.extended_clf_report()  # NOQA
    print('vamp roc = {}'.format(vamp_res.roc_score()))
    return vamp_res

def main():
    parser = _auto_argparse(fit)
    args, unknown = parser.parse_known_args()
    ns = args.__dict__.copy()
    fit(**ns)

if __name__ == '__main__':
    main()
