# print_model()


from raptor_hambla28 import h28_get_siamese_model, ori_get_siamese_model


# req_size=(105,105,1)
# model=ori_get_siamese_model(req_size,True)
# model.summary()

req_size=(498,280,3)
model=h28_get_siamese_model(req_size,True)
model.summary()
