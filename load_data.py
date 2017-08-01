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

        # Number of labels/products/categories
        self.n_cats = np.shape(self.products)[0]
        self.batch_counter = 0

    def next_batch(self):
        """ Load training batches"""
        # Get all user id's in train/test
        user_ids = self.orders[self.orders.eval_set == "train"].user_id
        # Number of unique user id's in train datapoints
        n_users = np.shape(user_ids)[0]
        # Initialize sparse data matrix
        X = np.zeros((self.batch_size, self.n_cats))
        y = np.zeros((self.batch_size, self.n_cats))

        bs = self.batch_size
        bc = self.batch_counter
        # For each user in the training set get all order ids
        for ii, uid in enumerate(user_ids[bc*bs:(bc+1)*bs]):
            # For each order_id get all product_id's and add to sparse matrix
            oids_tr = self.orders.loc[(self.orders.user_id == uid) &
                                      (self.orders.eval_set == "train")]
            oids_pr = self.orders.loc[(self.orders.user_id == uid) &
                                      (self.orders.eval_set == "prior")]
            # If order_id is in 'prior', add to data
            for oid in oids_pr.order_id.values:
                pids = self.priors[self.priors.order_id == oid].product_id
                for pid in pids.values:
                    X[ii, pid] += 1
            # If order_id is in 'train', add to labels
            oid = oids_tr.order_id.values[0]
            pids = self.priors[self.priors.order_id == oid].product_id
            for pid in pids.values:
                y[ii, pid] += 1
        self.batch_counter += 1
        if self.batch_counter == int(n_users/self.batch_size):
            self.batch_counter = 0
        return X, y

    def load_test(self):
        """ Load test set """
        # Get all user id's in train/test
        user_ids = self.orders[self.orders.eval_set == "test"].user_id[:10]
        # Number of unique user id's in train datapoints
        n_users = np.shape(user_ids)[0]

        X = np.zeros((n_users, self.n_cats))
        oids_to_predict = []
        for ii, uid in enumerate(user_ids):
            # For each order_id get all product_id's and add to sparse matrix
            oids_te = self.orders.loc[(self.orders.user_id == uid) &
                                      (self.orders.eval_set == "test")]
            oids_pr = self.orders.loc[(self.orders.user_id == uid) &
                                      (self.orders.eval_set == "prior")]
            for oid in oids_pr.order_id.values:
                # If order_id is in 'prior', add to data
                pids = self.priors[self.priors.order_id == oid].product_id
                for pid in pids.values:
                    X[ii, pid] += 1
            oids_to_predict.append(oids_te.order_id.values[0])
        return X, oids_to_predict
