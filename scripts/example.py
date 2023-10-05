import lpt4py as lpt

# create grid object
grid = lpt.Grid(N=512)

# create white noise
wn = grid.generate_noise(seed=12345)

print('wn shape:',wn.shape)
print('wn mean:',wn.mean())