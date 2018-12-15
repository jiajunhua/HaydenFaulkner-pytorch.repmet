

class TensorBoard(object):

    def __init__(self, every, tb_sw):
        self.every = every
        self.tb_sw = tb_sw

    def __call__(self, epoch, batch, step, model, stats):

        if batch % self.every == 0:

            # params = param[3]['self'].get_params()[0]  # pull params onto cpu (into the arg_params variable (param[3]['self'] is the module)
            # for k,v in params.items():
            #     if k[:3] == 'fc_' or k[:3] == 'emb' or k[:3] == 'rpn' or k[:3] == 'bbo' or k[:3] == 'cls' or k[:3] == 'rep':
            #         try:
            #             self.tb_log.add_histogram(tag=k,
            #                                          values=v.asnumpy(),
            #                                          bins=100,
            #                                          global_step=self.iter)
            #         except ValueError:
            #             print("ValueError: range parameter must be finite: %s min: %f  max: %f" % (str(k), np.min(v.asnumpy()), np.max(v.asnumpy())))
            #             print("You should really consider stopping training...")
            # if param.eval_metric is not None:
            #     name, value = param.eval_metric.get()
            for k, v in stats.items():
                self.tb_sw.add_scalar(tag=k, scalar_value=v, global_step=step)
