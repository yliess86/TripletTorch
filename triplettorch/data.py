"""Data module from triplettorch

The file provides a simple implementation of a TripletDataset. It can be used
to parkour datasets and provides a way to return multiple samples per label
to is the triplet miner work and give it more chance to find useful triplet
combinations.
"""
import numpy as np
import torch

from torch.utils.data import Dataset
from typing import Callable
from typing import Tuple

class TripletDataset( Dataset ):
    """TripletDataset

    The TripletDataset extends the standard Dataset provided by the pytorch
    utils. It provides simple access to data with the possibility of returning
    more than one sample per index based on the label.

    Attributes
    ----------
    labels  : np.ndarray
              Array containing all the labels respectively to each data sample.
              Labels needs to provide a way to access a sample label by index.
    data_fn : Callable
              The data_fn provides access to sample data given its index in the
              dataset. Providding a function instead of array has been chosen
              for preprocessing and other reasons.
    size    : int
              Size gives the dataset size, number of samples.
    n_sample: int
              The value represents the number of sample per index. The other
              samples will be chosen to be the same as the selected one. This
              allows to augment the number of possible valid triplet when used
              with a tripelt mining strategy.
    """

    def __init__(
        self    : 'TripletDataset',
        labels  : np.ndarray,
        data_fn : Callable,
        size    : int,
        n_sample: int
    ) -> None:
        """Init

        Parameters
        ----------
        labels  : np.ndarray
                  Array containing all the labels respectively to each data
                  sample. Labels needs to provide a way to access a sample label
                  by index.
        data_fn : Callable
                  The data_fn provides access to sample data given its index in
                  the dataset. Providding a function instead of array has been
                  chosen for preprocessing and other reasons.
        size    : int
                  Size gives the dataset size, number of samples.
        n_sample: int
                  The value represents the number of sample per index. The other
                  samples will be chosen to be the same as the selected one.
                  This allows to augment the number of possible valid triplet
                  when used with a tripelt mining strategy.
        """
        super( Dataset, self ).__init__( )
        self.labels   = labels
        self.data_fn  = data_fn
        self.size     = size
        self.n_sample = n_sample

    def __len__( self: 'TripletDataset' ) -> int:
        """Len

        Returns
        -------
        size: int
              Returns the size of the dataset, number of samples.
        """
        return self.size

    def __getitem__( self: 'TripletDataset', index: int ) -> Tuple[ np.ndarray ]:
        """GetItem

        Parameters
        ----------
        index: int
               Index of the sample to draw. The value should be less than the
               dataset size and positive.

        Returns
        -------
        labels: torch.Tensor
                Returns the labels respectively to each of the samples drawn.
                First sample is the sample is the one at the selected index,
                and others are selected randomly from the rest of the dataset.
        data  : torch.Tensor
                Returns the data respectively to each of the samples drawn.
                First sample is the sample is the one at the selected index,
                and others are selected randomly from the rest of the dataset.

        Raises
        ------
        IndexError: If index is negative or greater than the dataset size.
        """
        if not ( index >= 0 and index < len( self ) ):
            raise IndexError( f'Index { index } is out of range [ 0, { len( self ) } ]' )

        label         = np.array( [  self.labels[ index ] ] )
        datum         = np.array( [ self.data_fn( index ) ] )

        if self.n_sample == 1:
            return label, datum

        mask          = self.labels == label
        mask[ index ] = False
        mask          = mask.astype( np.float32 )

        indexes       = mask.nonzero( )[ 0 ]
        indexes       = np.random.choice( indexes, self.n_sample - 1, False )
        data          = np.array( [  self.data_fn( i ) for i in indexes ] )

        labels        = np.repeat( label, self.n_sample )
        data          = np.concatenate( ( datum, data ), axis = 0 )

        labels        = torch.from_numpy( labels )
        data          = torch.from_numpy(   data )

        return labels, data
