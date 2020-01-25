import math
import torch


class Affine():
    num_params = 2

    @staticmethod
    def get_pseudo_params(nn_outp):
        a = nn_outp[..., 0]  # [B, D]
        var_outp = nn_outp[..., 1]

        b = torch.exp(0.5 * var_outp)
        logbsq = var_outp

        return a, logbsq, b

    @staticmethod
    def standard(x, nn_outp):
        a, logbsq, b = Affine.get_pseudo_params(nn_outp)
        y = a + b * x
        logdet = 0.5 * logbsq.sum(-1)

        return y, logdet

    @staticmethod
    def reverse(y, nn_outp):
        a, logbsq, b = Affine.get_pseudo_params(nn_outp)

        x = (y - a) / b
        logdet = 0.5 * logbsq.sum(-1)

        return x, logdet


def arccosh(x):
    return torch.log(x + torch.sqrt(x.pow(2) - 1))


def arcsinh(x):
    return torch.log(x + torch.sqrt(x.pow(2) + 1))


class NLSq():
    num_params = 5
    logA = math.log(8 * math.sqrt(3) / 9 - 0.05)  # 0.05 is a small number to prevent exactly 0 slope

    @staticmethod
    def get_pseudo_params(nn_outp):

        # print(nn_outp.shape)
        # exit()

        a = nn_outp[..., 0]  # [B, D]
        logb = nn_outp[..., 1] * 0.4
        B = nn_outp[..., 2] * 0.3
        logd = nn_outp[..., 3] * 0.4
        f = nn_outp[..., 4]

        assert not (True in torch.isnan(logb)), "logb exploded"

        b = torch.exp(logb)
        d = torch.exp(logd)
        c = torch.tanh(B) * torch.exp(NLSq.logA + logb - logd)

        return a, b, c, d, f

    @staticmethod
    def standard(x, nn_outp):
        a, b, c, d, f = NLSq.get_pseudo_params(nn_outp)

        # print(a.shape)
        # print(nn_outp.shape)
        # exit()

        # double needed for stability. No effect on overall speed
        a = a.double()
        b = b.double()
        c = c.double()
        d = d.double()
        f = f.double()
        x = x.double()

        aa = -b * d.pow(2)
        bb = (x - a) * d.pow(2) - 2 * b * d * f
        cc = (x - a) * 2 * d * f - b * (1 + f.pow(2))
        dd = (x - a) * (1 + f.pow(2)) - c

        p = (3 * aa * cc - bb.pow(2)) / (3 * aa.pow(2))
        q = (2 * bb.pow(3) - 9 * aa * bb * cc + 27 * aa.pow(2) * dd) / (27 * aa.pow(3))

        t = -2 * torch.abs(q) / q * torch.sqrt(torch.abs(p) / 3)
        inter_term1 = -3 * torch.abs(q) / (2 * p) * torch.sqrt(3 / torch.abs(p))
        inter_term2 = 1 / 3 * arccosh(torch.abs(inter_term1 - 1) + 1)
        t = t * torch.cosh(inter_term2)

        tpos = -2 * torch.sqrt(torch.abs(p) / 3)
        inter_term1 = 3 * q / (2 * p) * torch.sqrt(3 / torch.abs(p))
        inter_term2 = 1 / 3 * arcsinh(inter_term1)
        tpos = tpos * torch.sinh(inter_term2)

        t[p > 0] = tpos[p > 0]
        y = t - bb / (3 * aa)

        arg = d * y + f
        denom = 1 + arg.pow(2)

        x_new = a + b * y + c / denom


        tmp = b - 2 * c * d * arg / denom.pow(2)

        assert not (True in torch.isnan(b)), "b exploded"
        assert not (True in torch.isnan(c)), "c exploded"
        assert not (True in torch.isnan(d)), "d exploded"
        assert not (True in torch.isnan(arg)), "arg exploded"
        assert not (True in torch.isnan(denom)), "demon exploded"
        assert not (0 in denom.pow(2)), "denom.pow(2) exploded"
        assert not(True in torch.isnan(tmp)), "tmp exploded"
        assert not(False in tmp > 0), "log(x), x < 0"


        logdet = -torch.log(b - 2 * c * d * arg / denom.pow(2) ).sum(-1)


        assert not(True in torch.isnan(logdet)), "log exploded"

        y = y.float()
        logdet = logdet.float()

        return y, logdet

    @staticmethod
    def reverse(y, nn_outp):
        a, b, c, d, f = NLSq.get_pseudo_params(nn_outp)

        arg = d * y + f
        denom = 1 + arg.pow(2)
        x = a + b * y + c / denom

        logdet = -torch.log(b - 2 * c * d * arg / denom.pow(2)).sum(-1)

        return x, logdet