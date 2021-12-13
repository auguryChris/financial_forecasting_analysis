import backtrader as bt
import datetime
import pandas as pd

class BasicStrategy(bt.Strategy):
    def __init__(self):
        # To keep track of pending orders
        self.order = None
        print('***The limit sells for this strategy are valid for 1 day***')
        
    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        elif order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)
            self.bar_executed = len(self)
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        # set no pending order
        self.order = None


    def next(self):
        # STRATEGY 
        data = self.datas[0]
        self.log(f'Current Portfolio Value : {self.broker.get_value()}')
        
        # cancel if there is an order pending, this strategy should have 1 working order per day
        if self.order:
            self.log('ORDER CANCELLED')
            self.cancel(self.order)
        
        # Check if we are in the market
        if not self.position:
            # BUY
            try:
                self.size = int(self.broker.get_cash() / self.datas[0].open[1])
            except:
                print('Size Exception. If at the end of data, ignore.')
            # get in position ASAP
            self.log(f'MARKET BUY CREATE {self.size} shares at next open, current close price: {data.close[0]}')
            self.buy(size=self.size) # market order buys at next open                      
                
        else:
            # place sell order at predicted high if predicted high is greater than current close price
            # TODO: Make prediction and close a filter in the class constructor (more optimal)
            if data.prediction[0] >= data.close[0]:
                self.log(f'LIMIT SELL CREATE {self.size} shares at {data.prediction[0]}')  
                self.sell(exectype=bt.Order.Limit,
                             price=data.prediction[0],
                             valid=data.datetime.date(0) + datetime.timedelta(days=2),
                             size=self.size)
        # if prediction is less than current value sell at open (ASAP)
            else:
                self.log('MARKET SELL CREATE. PREDICTION < CURRENT CLOSE')
                self.sell(size=self.size)
                

class BuyAndHold(bt.Strategy):
    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
        
    def start(self):
        # Track orders with this, dict must be named order_statistics_dict
        order_statistics_dict = {'timestamp':[], 'open':[], 'high':[],
                                    'low':[], 'close':[], 'volume':[], 'buy':[], 'sell':[]}

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)

            self.bar_executed = len(self)
      
    def start(self):
        self.val_start = self.broker.get_cash()  # keep the starting cash

    def nextstart(self):
        # Buy all the available cash
        size = int(self.broker.get_cash() / self.datas[0].open[1])
        self.buy(size=size)

    def stop(self):
        # calculate the actual returns
        self.roi = (self.broker.get_value() / self.val_start) - 1.0
        print(f'Stop price: {self.datas[0].close[0]}')
        print('ROI:        {:.2f}%'.format(100.0 * self.roi))

def prepare_data(data, fromdate, todate, filepath):
    """Prepare data for backtrader datafeed object
        Returns prepared data filepath and params for the GenericCSVData class, also returns columns used"""
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    # start param setup for backtrader
    start = data['timestamp'].iloc[0]
    end = data['timestamp'].iloc[-1]
    from_to = [(start.year, start.month, start.day), (end.year, end.month, end.day)]
    # Backtrader string format
    data['timestamp'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # backtrader data feed class needs a file path, this could be a temp file that's constantly overwritten
    data.to_csv(filepath)
    
    starting_params = [
    ('fromdate', datetime.datetime(*from_to[0])),
    ('todate', datetime.datetime(*from_to[1])),
    ('nullvalue', 0.0),
    ('dtformat', ('%Y-%m-%d %H:%M:%S')),
    ('tmformat', ('%H:%M:%S')),
    ('datetime', 1)]
    # skip nonfeatures (timestamp)
    cols = data.columns[1:]
    # get column position for each indicator and add to starting params list
    i = 2 # starting index since others are reserved
    for indicator in cols:
        starting_params.append((indicator, i))
        i+=1
    final_params = tuple(starting_params)

    return filepath, final_params, cols 