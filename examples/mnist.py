import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch

from triplettorch import HardNegativeTripletMiner
from triplettorch import AllTripletMiner
from torch.utils.data import DataLoader
from triplettorch import TripletDataset
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm

# Convolutionnal Network
# Input : ( None, 1, 28, 28 )
# Output: ( None, 2 )
class Model( nn.Module ):
    def __init__( self: 'Model' ) -> None:
        super( Model, self ).__init__( )
        self.conv1 = nn.Conv2d(  1, 20, 5, 1 )
        self.conv2 = nn.Conv2d( 20, 50, 5, 1 )
        self.fc1   = nn.Linear( 4 * 4 * 50, 512 )
        self.fc2   = nn.Linear(        512,   2 )

    def forward( self: 'Model', X: torch.Tensor ) -> torch.Tensor:
        X = torch.relu( self.conv1( X ) )
        X = torch.max_pool2d( X, 2, 2 )
        X = torch.relu( self.conv2( X ) )
        X = torch.max_pool2d( X, 2, 2 )
        X = X.view( -1, 4 * 4 * 50 )
        X = torch.relu( self.fc1( X ) )
        X = self.fc2( X )
        return X

# Import Data with normalization
train_set      = datasets.MNIST(
    './data',
    train     = True,
    download  = True,
    transform = transforms.Compose( [
        transforms.ToTensor( ),
        transforms.Normalize( ( .1307, ), ( .3081 ) )
    ] )
)
test_set       = datasets.MNIST(
    './data',
    train     = False,
    download  = True,
    transform = transforms.Compose( [
        transforms.ToTensor( ),
        transforms.Normalize( ( .1307, ), ( .3081 ) )
    ] )
)

# Create Lambda function to access data in correct format
train_set_d    = lambda index: train_set.data[ index ].unsqueeze( 0 ).float( ).numpy( )
test_set_d     = lambda index:  test_set.data[ index ].unsqueeze( 0 ).float( ).numpy( )

# Hyperparameters
epochs         = 200
batch_size     = 64
n_sample       = 6

# Triplet Dataset Definition
tri_train_set  = TripletDataset( train_set.targets.numpy( ), train_set_d, train_set.targets.size( 0 ), n_sample )
tri_test_set   = TripletDataset(  test_set.targets.numpy( ),  test_set_d,  test_set.targets.size( 0 ),        1 )

# Data Loader
tri_train_load = DataLoader( tri_train_set,
    batch_size  = batch_size,
    shuffle     = True,
    num_workers = 2,
    pin_memory  = True
)
tri_test_load  = DataLoader( tri_test_set,
    batch_size  = batch_size,
    shuffle     = False,
    num_workers = 2,
    pin_memory  = True
)

# Model Loss Optimizer
model          = Model( ).cuda( )
miner          = HardNegativeTripletMiner( .5 ).cuda( ) # AllTripletMiner( .5 ).cuda( )
optim          = optim.Adam( model.parameters( ), lr = 1e-3 )

# Figure for Embeddings plot
fig            = plt.figure( figsize = ( 8, 8 ) )
ax             = fig.add_subplot( 111 )

# Main loop
for e in tqdm( range( epochs ), desc = 'Epoch' ):
    # ================== TRAIN ========================
    train_n        = len( tri_train_load )
    train_loss     = 0.
    train_frac_pos = 0.

    with tqdm( tri_train_load, desc = 'Batch' ) as b_pbar:
        for b, batch in enumerate( b_pbar ):
            optim.zero_grad( )

            labels, data    = batch
            labels          = torch.cat( [ label for label in labels ], axis = 0 )
            data            = torch.cat( [ datum for datum in   data ], axis = 0 )
            labels          = labels.cuda( )
            data            = data.cuda( )

            embeddings      = model( data )
            loss, frac_pos  = miner( labels, embeddings )

            loss.backward( )
            optim.step( )

            train_loss     += loss.detach( ).item( )
            train_frac_pos += frac_pos.detach( ).item( ) if frac_pos is not None else \
                              0.

            b_pbar.set_postfix(
                train_loss     = train_loss / train_n,
                train_frac_pos = f'{( train_frac_pos / train_n ):.2%}'
            )

    # ================== TEST ========================
    if e % 5 == 0 and e > 0:
        test_embeddings = [ ]
        test_labels     = [ ]

        for b, batch in enumerate( tqdm( tri_test_load, desc = 'Plot' ) ):
            labels, data = batch
            data         = torch.cat( [ datum for datum in   data ], axis = 0 )
            labels       = torch.cat( [ label for label in labels ], axis = 0 )
            embeddings   = model( data.cuda( ) ).detach( ).cpu( ).numpy( )
            labels       = labels.numpy( )

            test_embeddings.append( embeddings )
            test_labels.append( labels )

        test_embeddings = np.concatenate( test_embeddings, axis = 0 )
        test_labels     = np.concatenate(     test_labels, axis = 0 )

        ax.clear( )
        ax.scatter(
            test_embeddings[ :, 0 ],
            test_embeddings[ :, 1 ],
            c = test_labels
        )
        fig.canvas.draw( )
        fig.savefig( f'./MNIST{ miner.__class__ }.png' )
