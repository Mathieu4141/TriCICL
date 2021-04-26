from tricicl.triplet_loss.trainer import TripletLossTrainerPlugin


class TripletLossPostTrainingPlugin(TripletLossTrainerPlugin):
    after_training_exp = TripletLossTrainerPlugin.train
