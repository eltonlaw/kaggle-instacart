""" Load data """
# pylint: disable=invalid-name,line-too-long,too-few-public-methods,too-many-instance-attributes,no-member
import numpy as np
import pandas as pd

class Instacart:
    """ Instacart dataset preprocesser and batch loader"""
    def __init__(self, batch_size=13, path="data/"):
        self.batch_size = batch_size
        self.train = pd.read_csv(path+'order_products__train.csv',
                                 dtype={
                                     'order_id': np.int32,
                                     'product_id': np.uint16,
                                     'add_to_cart_order': np.uint8,
                                     'reordered': np.int8
                                 })

        self.priors = pd.read_csv(path+'order_products__prior.csv',
                                  dtype={
                                      'order_id': np.int32,
                                      'product_id': np.uint16,
                                      'add_to_cart_order': np.uint8,
                                      'reordered': np.int8
                                  })

        self.orders = pd.read_csv(path+'orders.csv',
                                  dtype={
                                      'order_id': np.uint32,
                                      'user_id': np.uint32,
                                      'order_number': np.uint8,
                                      'order_dow': np.uint8,
                                      'order_hour_of_day': np.uint8,
                                      'days_since_prior_order': np.float16
                                  })
        self.products = pd.read_csv(path+'products.csv',
                                    dtype={
                                        'product_id': np.uint16,
                                        'product_name': str,
                                        'aisle_id': np.uint8,
                                        'department_id': np.uint8
                                    })
        # Get all unique user id's in train
        self.user_ids = self.train["order_id"].unique()
        # Number of labels/products/categories
        self.n_cats = np.shape(self.products)[0]
        # Number of unique user id's in train/training datapoints
        self.n_users = np.shape(self.user_ids)[0]
        self.batch_counter = 0

    def next_batch(self):
        """ Load batches"""
        # Initialize sparse data matrix
        data = np.zeros((self.batch_size, self.n_cats))

        bs = self.batch_size
        bc = self.batch_counter
        # For each user in the training set get all order ids
        for ii, uid in enumerate(self.user_ids[bc*bs:(bc+1)*bs]):
            oids = self.orders.loc[self.orders['user_id'] == uid]["order_id"]

            # For each order_id get all product_id's and add to sparse matrix
            for oid in oids.values:
                pids = self.priors[self.priors["order_id"] == oid]["product_id"]
                for pid in pids.values:
                    data[ii, pid] += 1
        self.batch_counter += 1
        return data
