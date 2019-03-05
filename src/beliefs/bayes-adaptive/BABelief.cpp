#include "BABelief.hpp"

#include "beliefs/bayes-adaptive/BAImportanceSampling.hpp"
#include "beliefs/bayes-adaptive/BAPointEstimation.hpp"
#include "beliefs/bayes-adaptive/BARejectionSampling.hpp"
#include "beliefs/bayes-adaptive/NestedBelief.hpp"
#include "beliefs/bayes-adaptive/factored/MHNIPS2018.hpp"
#include "beliefs/bayes-adaptive/factored/MHwithinGibbs.hpp"
#include "beliefs/bayes-adaptive/factored/ReinvigoratingRejectionSampling.hpp"
#include "beliefs/bayes-adaptive/factored/StructureIncubatorSampling.hpp"
#include "beliefs/bayes-adaptive/prototypes/CheatingReinvigoration.hpp"
#include "configurations/Conf.hpp"

namespace factory {

std::unique_ptr<beliefs::BABelief> makeBABelief(configurations::Conf const& c)
{

    if (c.belief == "point_estimate")
        return std::unique_ptr<beliefs::BABelief>(new beliefs::BAPointEstimation());
    if (c.belief == "rejection_sampling")
        return std::unique_ptr<beliefs::BABelief>(
            new beliefs::BARejectionSampling(c.belief_conf.particle_amount));
    if (c.belief == "importance_sampling")
        return std::unique_ptr<beliefs::BABelief>(
            new beliefs::BAImportanceSampling(c.belief_conf.particle_amount));

    if (c.belief == "reinvigoration")
        return std::unique_ptr<beliefs::BABelief>(
            new beliefs::bayes_adaptive::factored::ReinvigoratingRejectionSampling(
                c.belief_conf.particle_amount, c.belief_conf.resample_amount));

    if (c.belief == "mh-nips")
        return std::unique_ptr<beliefs::BABelief>(new beliefs::bayes_adaptive::factored::MHNIPS2018(
            c.belief_conf.particle_amount, c.belief_conf.threshold));

    if (c.belief == "mh-within-gibbs")
    {
        if (c.belief_conf.option.empty())
            return std::unique_ptr<beliefs::BABelief>(
                new beliefs::bayes_adaptive::factored::MHwithinGibbs(
                    c.belief_conf.particle_amount,
                    c.belief_conf.threshold,
                    beliefs::bayes_adaptive::factored::MHwithinGibbs::MSG));
        else if (c.belief_conf.option == "rs")
            return std::unique_ptr<beliefs::BABelief>(
                new beliefs::bayes_adaptive::factored::MHwithinGibbs(
                    c.belief_conf.particle_amount,
                    c.belief_conf.threshold,
                    beliefs::bayes_adaptive::factored::MHwithinGibbs::RS));
    }

    if (c.belief == "incubator")
        return std::unique_ptr<beliefs::BABelief>(
            new beliefs::bayes_adaptive::factored::StructureIncubatorSampling(
                c.belief_conf.particle_amount,
                c.belief_conf.resample_amount,
                c.belief_conf.threshold));

    if (c.belief == "cheating-reinvigoration")
        return std::unique_ptr<beliefs::BABelief>(
            new beliefs::bayes_adaptive::prototypes::CheatingReinvigoration(
                c.belief_conf.particle_amount,
                c.belief_conf.resample_amount,
                c.belief_conf.threshold));

    if (c.belief == "nested")
        return std::unique_ptr<beliefs::BABelief>(new beliefs::bayes_adaptive::NestedBelief(
            c.belief_conf.particle_amount,
            c.belief_conf.particle_amount * c.belief_conf.particle_amount));

    throw "incorrect state estimator provided";
}

} // namespace factory
