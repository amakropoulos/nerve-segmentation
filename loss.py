import theano
import theano.tensor as T

def dice(pred, tgt, ss=1):
    return -2*(T.sum(pred*tgt)+ss)/(T.sum(pred) + T.sum(tgt) + ss)

# # actual loss (Dice is not exactly continuous)
# def dice(pred, tgt):
#     predeq = (pred >= 0.5)
#     tgteq = (tgt >= 0.5)
#     den = predeq.sum() + tgteq.sum()
#     if den == 0: return -1
#     return -2* (predeq*tgteq).sum()/den


def squared_error(pred, tgt):
	import lasagne
	return lasagne.objectives.squared_error(pred, tgt).mean()
