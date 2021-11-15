#include "catch.hpp"

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "bayes-adaptive/priors/BAPOMDPPrior.hpp"
#include "bayes-adaptive/priors/FBAPOMDPPrior.hpp"
#include "bayes-adaptive/states/factored/FBAPOMDPState.hpp"
#include "configurations/FBAConf.hpp"
#include "domains/dummy/FactoredDummyDomain.hpp"
#include "domains/dummy/FactoredDummyDomainBAExtension.hpp"
#include "domains/dummy/FactoredDummyDomainFBAExtension.hpp"
#include "domains/tiger/FactoredTiger.hpp"
#include "domains/tiger/FactoredTigerBAExtension.hpp"
#include "domains/tiger/FactoredTigerFBAExtension.hpp"
#include "environment/State.hpp"
#include "utils/index.hpp"
#include "utils/random.hpp"

SCENARIO("fbapomdp state sampling", "[bayes-adaptive][factored]")
{
    auto up    = IndexAction(std::to_string(domains::FactoredDummyDomain::ACTIONS::UP)),
         right = IndexAction(std::to_string(domains::FactoredDummyDomain::ACTIONS::RIGHT));

    GIVEN("A Dummy FBAPOMDP State of size 3")
    {

        auto const size = 3;
        auto c          = configurations::FBAConf();

        c.domain_conf.size   = size;
        c.domain_conf.domain = "factored-dummy";

        auto const d     = domains::FactoredDummyDomain(3);
        auto const prior = factory::makeFBAPOMDPPrior(d, c);

        auto fba_state = prior->sample(d.sampleStartState());

        THEN("sampling a new state with up action should return (1,0) state")
        {
            REQUIRE(
                fba_state->sampleStateIndex(
                    fba_state->_domain_state, &up, rnd::sample::Dir::sampleFromSampledMult)
                == "1");
            REQUIRE(
                fba_state->sampleStateIndex(
                    fba_state->_domain_state, &up, rnd::sample::Dir::sampleFromExpectedMult)
                == "1");
        }

        THEN("sampling a new state with right action should return (0,1) state")
        {
            REQUIRE(
                fba_state->sampleStateIndex(
                    fba_state->_domain_state, &right, rnd::sample::Dir::sampleFromSampledMult)
                == "3");
            REQUIRE(
                fba_state->sampleStateIndex(
                    fba_state->_domain_state, &right, rnd::sample::Dir::sampleFromExpectedMult)
                == "3");
        }

        THEN("sampling an observation should always return 0")
        {
            REQUIRE(
                fba_state->sampleObservationIndex(
                    &up, fba_state->_domain_state, rnd::sample::Dir::sampleFromSampledMult)
                == 0);
            REQUIRE(
                fba_state->sampleObservationIndex(
                    &up, fba_state->_domain_state, rnd::sample::Dir::sampleFromExpectedMult)
                == 0);

            REQUIRE(
                fba_state->sampleObservationIndex(
                    &right, fba_state->_domain_state, rnd::sample::Dir::sampleFromSampledMult)
                == 0);
            REQUIRE(
                fba_state->sampleObservationIndex(
                    &right, fba_state->_domain_state, rnd::sample::Dir::sampleFromExpectedMult)
                == 0);
        }

        d.releaseState(fba_state->_domain_state);
        delete (fba_state);
    }
}

SCENARIO("fbapomdpstate initiation", "[bayes-adaptive][factored]")
{
    GIVEN("A random domain")
    {
        auto action_size = 4;
        auto state       = IndexState("0");

        auto domain_sizes  = Domain_Size(40, action_size, 6);
        auto feature_sizes = Domain_Feature_Size({2, 4, 5}, {2, 3});
        auto step_sizes    = bayes_adaptive::factored::BABNModel::Indexing_Steps(
            indexing::stepSize(feature_sizes._S), indexing::stepSize(feature_sizes._O));

        WHEN("a factored state with given domain size is initiated disconnectedly")
        {

            auto const model =
                bayes_adaptive::factored::BABNModel(&domain_sizes, &feature_sizes, &step_sizes);

            THEN(
                "we hope the corresponding nodes have the correct output size & number of "
                "paramters")
            {
                for (auto a = 0; a < action_size; ++a)
                {
                    auto action = IndexAction(std::to_string(a));
                    REQUIRE(model.transitionNode(&action, 0).range() == 2);
                    REQUIRE(model.transitionNode(&action, 0).numParams() == 2);

                    REQUIRE(model.transitionNode(&action, 1).range() == 4);
                    REQUIRE(model.transitionNode(&action, 1).numParams() == 4);

                    REQUIRE(model.transitionNode(&action, 2).range() == 5);
                    REQUIRE(model.transitionNode(&action, 2).numParams() == 5);

                    REQUIRE(model.observationNode(&action, 0).range() == 2);
                    REQUIRE(model.observationNode(&action, 0).numParams() == 2);

                    REQUIRE(model.observationNode(&action, 1).range() == 3);
                    REQUIRE(model.observationNode(&action, 1).numParams() == 3);
                }
            }
        }
    }
}

SCENARIO("factored dummy factored prior", "[factored][bayes-adaptive][dummy]")
{

    auto const size = 5;
    auto c          = configurations::FBAConf();

    c.domain_conf.size   = size;
    c.domain_conf.domain = "factored-dummy";

    auto const d   = domains::FactoredDummyDomain(size);
    auto const ext = bayes_adaptive::domain_extensions::FactoredDummyDomainBAExtension(size);
    auto const p   = factory::makeFBAPOMDPPrior(d, c);

    auto ba_state = p->sample(d.sampleStartState());

    auto a_up    = IndexAction(std::to_string(domains::FactoredDummyDomain::ACTIONS::UP)),
         a_right = IndexAction(std::to_string(domains::FactoredDummyDomain::ACTIONS::RIGHT));

    THEN("going up should sample a state higher")
    {
        REQUIRE(
            ba_state->sampleStateIndex(
                ba_state->_domain_state, &a_up, rnd::sample::Dir::sampleFromSampledMult)
            == "1");
        REQUIRE(
            ba_state->sampleStateIndex(
                ba_state->_domain_state, &a_up, rnd::sample::Dir::sampleFromExpectedMult)
            == "1");

        d.releaseState(ba_state->_domain_state);
        ba_state->_domain_state = ext.getState("1");

        AND_THEN("another step up should do the same")
        {
            REQUIRE(
                ba_state->sampleStateIndex(
                    ba_state->_domain_state, &a_up, rnd::sample::Dir::sampleFromSampledMult)
                == "2");
            REQUIRE(
                ba_state->sampleStateIndex(
                    ba_state->_domain_state, &a_up, rnd::sample::Dir::sampleFromExpectedMult)
                == "2");
        }

        AND_THEN("a step right here should increase by step size")
        {
            REQUIRE(
                ba_state->sampleStateIndex(
                    ba_state->_domain_state, &a_right, rnd::sample::Dir::sampleFromSampledMult)
                == std::to_string(1 + size));
            REQUIRE(
                ba_state->sampleStateIndex(
                    ba_state->_domain_state, &a_right, rnd::sample::Dir::sampleFromExpectedMult)
                == std::to_string(1 + size));
        }
    }

    THEN("going right should sample a state to the right")
    {
        REQUIRE(
            ba_state->sampleStateIndex(
                ba_state->_domain_state, &a_right, rnd::sample::Dir::sampleFromSampledMult)
            == std::to_string(size));
        REQUIRE(
            ba_state->sampleStateIndex(
                ba_state->_domain_state, &a_right, rnd::sample::Dir::sampleFromExpectedMult)
            == std::to_string(size));

        d.releaseState(ba_state->_domain_state);
        ba_state->_domain_state = ext.getState(std::to_string(size));

        AND_THEN("another step up should do the same")
        {
            REQUIRE(
                ba_state->sampleStateIndex(
                    ba_state->_domain_state, &a_right, rnd::sample::Dir::sampleFromSampledMult)
                == std::to_string(2 * size));
            REQUIRE(
                ba_state->sampleStateIndex(
                    ba_state->_domain_state, &a_right, rnd::sample::Dir::sampleFromExpectedMult)
                == std::to_string(2 * size));
        }

        AND_THEN("a step up here should increase by 1")
        {
            REQUIRE(
                ba_state->sampleStateIndex(
                    ba_state->_domain_state, &a_up, rnd::sample::Dir::sampleFromSampledMult)
                == std::to_string(1 + size));
            REQUIRE(
                ba_state->sampleStateIndex(
                    ba_state->_domain_state, &a_up, rnd::sample::Dir::sampleFromExpectedMult)
                == std::to_string(1 + size));
        }
    }

    d.releaseState(ba_state->_domain_state);
    delete (ba_state);
}

SCENARIO("copy fbapomdp states", "[bayes-adaptive][flat]")
{
    GIVEN("A BAPOMDPState copied from another")
    {
        auto c               = configurations::FBAConf();
        c.domain_conf.domain = "factored-dummy";
        c.domain_conf.size   = 5;

        auto const d = domains::FactoredDummyDomain(c.domain_conf.size);
        auto const ext =
            bayes_adaptive::domain_extensions::FactoredDummyDomainBAExtension(c.domain_conf.size);
        auto const p = factory::makeTBAPOMDPPrior(d, c);

        auto s1 = p->sample(ext.getState("0"));
        auto s2 = s1->copy(d.copyState(s1->_domain_state));

        WHEN("We update (change) one of the copies")
        {
            d.releaseState(s2->_domain_state);
            s2->_domain_state = ext.getState("1");

            THEN("It should be different from the other copy")
            {
                REQUIRE(s1->_domain_state != s2->_domain_state);
                REQUIRE(s1->_domain_state->index() == "0");
                REQUIRE(s2->_domain_state->index() == "1");
            }
        }

        d.releaseState(s1->_domain_state);
        d.releaseState(s2->_domain_state);

        delete (s1);
        delete (s2);
    }
}

SCENARIO("compute fbapomdp observation probabilities", "[domain][factored][bayes-adaptive][dummy]")
{
    auto conf = configurations::FBAConf();

    conf.domain_conf.size = 5;

    GIVEN("a fbapomdp state in the factored dummy domain of size 5")
    {
        conf.domain_conf.domain = "factored-dummy";

        auto const d   = domains::FactoredDummyDomain(conf.domain_conf.size);
        auto const ext = bayes_adaptive::domain_extensions::FactoredDummyDomainBAExtension(
            conf.domain_conf.size);
        auto const p = factory::makeFBAPOMDPPrior(d, conf);

        auto ba_state = p->sample(d.sampleStartState());

        auto a_up    = IndexAction(std::to_string(domains::FactoredDummyDomain::ACTIONS::UP)),
             a_right = IndexAction(std::to_string(domains::FactoredDummyDomain::ACTIONS::RIGHT));

        auto o = IndexObservation("0");

        WHEN("we calculate the probability of an observation")
        {
            auto s = ext.getState(std::to_string(rnd::slowRandomInt(0, ext.domainSize()._S)));

            REQUIRE(
                ba_state->computeObservationProbability(&o, &a_up, s, rnd::sample::Dir::sampleMult)
                == 1);
            REQUIRE(
                ba_state->computeObservationProbability(
                    &o, &a_up, s, rnd::sample::Dir::expectedMult)
                == 1);
            REQUIRE(
                ba_state->computeObservationProbability(&o, &a_up, s, rnd::sample::Dir::sampleMult)
                == 1);
            REQUIRE(
                ba_state->computeObservationProbability(
                    &o, &a_up, s, rnd::sample::Dir::expectedMult)
                == 1);

            d.releaseState(s);
        }

        d.releaseState(ba_state->_domain_state);
        delete (static_cast<FBAPOMDPState*>(ba_state));
    }

    GIVEN("a fbapomdp state in the factored tiger domain")
    {
        conf.domain_conf.domain = "continuous-factored-tiger";

        auto const d = domains::FactoredTiger(
            domains::FactoredTiger::FactoredTigerDomainType::CONTINUOUS, conf.domain_conf.size);
        auto const p = factory::makeFBAPOMDPPrior(d, conf);

        auto ba_state = p->sample(d.sampleStartState());

        WHEN("we compute the probability of an observation")
        {
            auto listen    = IndexAction(std::to_string(domains::FactoredTiger::OBSERVE)),
                 open_door = IndexAction(std::to_string(rnd::slowRandomInt(0, 2)));

            auto s = d.sampleStartState();

            auto correct_ob = IndexObservation(std::to_string(d.tigerLocation(s))),
                 incorrt_ob = IndexObservation(std::to_string(1 - std::stoi(correct_ob.index())));

            THEN("Listening produces the dfeault .85/.15 ratio")
            {
                REQUIRE(
                    ba_state->computeObservationProbability(
                        &correct_ob, &listen, s, rnd::sample::Dir::sampleMult)
                    == Approx(.85).epsilon(.01));
                REQUIRE(
                    ba_state->computeObservationProbability(
                        &correct_ob, &listen, s, rnd::sample::Dir::expectedMult)
                    == Approx(.85).epsilon(.01));

                REQUIRE(
                    ba_state->computeObservationProbability(
                        &incorrt_ob, &listen, s, rnd::sample::Dir::sampleMult)
                    == Approx(.15).epsilon(.01));
                REQUIRE(
                    ba_state->computeObservationProbability(
                        &incorrt_ob, &listen, s, rnd::sample::Dir::expectedMult)
                    == Approx(.15).epsilon(.01));
            }

            THEN("Opening doors is always 50/50")
            {
                REQUIRE(
                    ba_state->computeObservationProbability(
                        &correct_ob, &open_door, s, rnd::sample::Dir::sampleMult)
                    == Approx(.5).epsilon(.01));
                REQUIRE(
                    ba_state->computeObservationProbability(
                        &correct_ob, &open_door, s, rnd::sample::Dir::expectedMult)
                    == Approx(.5).epsilon(.01));

                REQUIRE(
                    ba_state->computeObservationProbability(
                        &incorrt_ob, &open_door, s, rnd::sample::Dir::sampleMult)
                    == Approx(.5).epsilon(.01));
                REQUIRE(
                    ba_state->computeObservationProbability(
                        &incorrt_ob, &open_door, s, rnd::sample::Dir::expectedMult)
                    == Approx(.5).epsilon(.01));
            }

            d.releaseState(s);
        }

        d.releaseState(ba_state->_domain_state);
        delete (static_cast<FBAPOMDPState*>(ba_state));
    }
}
