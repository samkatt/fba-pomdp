#include "catch.hpp"

#include <string>

#include "bayes-adaptive/priors/BAPOMDPPrior.hpp"
#include "bayes-adaptive/states/table/BAPOMDPState.hpp"
#include "configurations/BAConf.hpp"
#include "domains/dummy/DummyDomain.hpp"
#include "domains/dummy/DummyDomainBAExtension.hpp"
#include "domains/tiger/Tiger.hpp"
#include "domains/tiger/TigerBAExtension.hpp"
#include "utils/random.hpp"

SCENARIO("sample from BAPODMP states", "[bayes-adaptive][flat]")
{
    using rnd::sample::Dir::SAMPLETYPE;
    for (auto method : {SAMPLETYPE::Regular, SAMPLETYPE::Expected})
    {

        WHEN("using sampling method: " + std::to_string(method))
        {
            configurations::BAConf c;

            c.bayes_sample_method = method;

            auto m = (method == 0) ? rnd::sample::Dir::sampleFromSampledMult
                                   : rnd::sample::Dir::sampleFromExpectedMult;

            GIVEN("Dummy domain")
            {

                c.domain_conf.domain = "dummy";

                auto const d   = domains::DummyDomain();
                auto const ext = bayes_adaptive::domain_extensions::DummyDomainBAExtension();
                auto const p   = factory::makeTBAPOMDPPrior(d, c);

                auto ba_state = static_cast<BAPOMDPState*>(p->sample(d.sampleStartState()));

                THEN("Sampling states and observations always return 0")
                {
                    auto s = IndexState(0);
                    auto a = IndexAction(std::to_string(0));

                    for (auto i = 0; i < 10; ++i)
                    {
                        REQUIRE(ba_state->sampleStateIndex(&s, &a, m) == 0);
                        REQUIRE(ba_state->sampleObservationIndex(&a, &s, m) == 0);
                    }
                }

                d.releaseState(ba_state->_domain_state);
                delete (ba_state);
            }

            GIVEN("Tiger domain")
            {

                c.domain_conf.domain = "episodic-tiger";

                auto const d   = domains::Tiger(domains::Tiger::TigerType::EPISODIC);
                auto const ext = bayes_adaptive::domain_extensions::TigerBAExtension(
                    domains::Tiger::TigerType::EPISODIC);
                auto const p = factory::makeTBAPOMDPPrior(d, c);

                auto ba_state = static_cast<BAPOMDPState*>(p->sample(d.sampleStartState()));

                THEN("Sampling states and observations is always within domain size")
                {
                    for (auto i = 0; i < 10; ++i)
                    {
                        auto a     = d.generateRandomAction(ba_state->_domain_state);
                        auto new_s = d.sampleStartState();

                        REQUIRE(
                            ba_state->sampleStateIndex(ba_state->_domain_state, a, m)
                            < ext.domainSize()._S);
                        REQUIRE(
                            ba_state->sampleObservationIndex(a, new_s, m) < ext.domainSize()._O);

                        d.releaseState(new_s);
                        d.releaseAction(a);
                    }
                }

                THEN("Sampling states when observing returns the same state")
                {
                    auto listen = IndexAction(domains::Tiger::Literal::OBSERVE);

                    for (auto i = 0; i < 10; ++i)
                    {
                        // try for different initial states
                        d.releaseState(ba_state->_domain_state);
                        ba_state->_domain_state = d.sampleStartState();

                        REQUIRE(
                            ba_state->sampleStateIndex(ba_state->_domain_state, &listen, m)
                            == ba_state->index());
                    }
                }

                d.releaseState(ba_state->_domain_state);
                delete (ba_state);
            }
        }
    }
}

SCENARIO("updating BAPOMDPState", "[bayes-adaptive][flat]")
{
    GIVEN("a BAPOMDPState for the Tiger domain")
    {
        configurations::BAConf c;
        c.domain_conf.domain = "episodic-tiger";

        auto const d   = domains::Tiger(domains::Tiger::TigerType::EPISODIC);
        auto const ext = bayes_adaptive::domain_extensions::TigerBAExtension(
            domains::Tiger::TigerType::EPISODIC);
        auto const p = factory::makeTBAPOMDPPrior(d, c);

        auto ba_state = static_cast<BAPOMDPState*>(p->sample(d.sampleStartState()));

        THEN("Update a random transition should increment *only* that count")
        {
            for (int i = 0; i < 10; ++i)
            {
                auto a     = d.generateRandomAction(ba_state->_domain_state);
                auto s     = d.sampleStartState();
                auto new_s = d.copyState(s);

                Observation const* o(nullptr);
                Reward r(0);

                auto ba_state_copy =
                    static_cast<BAPOMDPState*>(ba_state->copy(d.sampleStartState()));

                // take step to generate random observation
                d.step(&new_s, a, &o, &r);

                ba_state_copy->incrementCountsOf(s, a, o, new_s);

                for (auto s_i = 0; s_i < ext.domainSize()._S; ++s_i)
                {
                    for (auto a_i = 0; a_i < ext.domainSize()._A; ++a_i)
                    {
                        for (auto o_i = 0; o_i < ext.domainSize()._O; ++o_i)
                        {
                            for (auto new_s_i = 0; new_s_i < ext.domainSize()._S; ++new_s_i)
                            {

                                auto state       = IndexState(s_i);
                                auto action      = IndexAction(a_i);
                                auto observation = IndexObservation(o_i);
                                auto new_state   = IndexState(new_s_i);

                                if (s_i !=std::stoi(s->index())|| a_i != std::stoi(a->index())
                                    || new_s_i !=std::stoi(new_s->index()))
                                {
                                    REQUIRE(
                                        ba_state->model()->count(&state, &action, &new_state)
                                        == Approx(ba_state_copy->model()->count(
                                               &state, &action, &new_state)));
                                } else
                                { // the exact update!!! should be incremented
                                    REQUIRE(
                                        ba_state->model()->count(&state, &action, &new_state)
                                        == Approx(
                                               ba_state_copy->model()->count(
                                                   &state, &action, &new_state)
                                               - 1));
                                }

                                if (a_i != std::stoi(a->index()) || o_i != std::stoi(o->index())
                                    || new_s_i !=std::stoi(new_s->index()))
                                {
                                    REQUIRE(
                                        ba_state->model()->count(&action, &new_state, &observation)
                                        == Approx(ba_state_copy->model()->count(
                                               &action, &new_state, &observation)));
                                } else
                                { // the exact update!!! should be incremented
                                    REQUIRE(
                                        ba_state->model()->count(&action, &new_state, &observation)
                                        == Approx(
                                               ba_state_copy->model()->count(
                                                   &action, &new_state, &observation)
                                               - 1));
                                }
                            }
                        }
                    }
                }

                d.releaseState(ba_state_copy->_domain_state);
                d.releaseAction(a);
                d.releaseState(s);
                d.releaseState(new_s);
                d.releaseObservation(o);
                delete (ba_state_copy);
            }
        }

        d.releaseState(ba_state->_domain_state);
        delete (ba_state);
    }
}

SCENARIO("copy bapomdp states", "[bayes-adaptive][flat]")
{
    GIVEN("A BAPOMDPState copied from another")
    {

        auto c               = configurations::BAConf();
        c.domain_conf.domain = "episodic-tiger";

        auto const d   = domains::Tiger(domains::Tiger::TigerType::EPISODIC);
        auto const ext = bayes_adaptive::domain_extensions::TigerBAExtension(
            domains::Tiger::TigerType::EPISODIC);
        auto const p = factory::makeTBAPOMDPPrior(d, c);

        auto s1 = p->sample(ext.getState(0));
        auto s2 = s1->copy(d.copyState(s1->_domain_state));

        WHEN("We update (change) one of the copies")
        {
            d.releaseState(s2->_domain_state);
            s2->_domain_state = ext.getState(1);

            THEN("It should be different from the other copy")
            {
                REQUIRE(s1->_domain_state != s2->_domain_state);
                REQUIRE(s1->_domain_state->index() == 0);
                REQUIRE(s2->_domain_state->index() == 1);
            }
        }

        d.releaseState(s1->_domain_state);
        d.releaseState(s2->_domain_state);

        delete (s1);
        delete (s2);
    }
}

SCENARIO("compute BAPOMDP observation probabilitieis", "[bayes-adaptive][flat][domain]")
{

    configurations::BAConf c;

    GIVEN("BAPOMDP state of the dummy domain")
    {
        c.domain_conf.domain = "dummy";

        auto const d = domains::DummyDomain();
        auto const p = factory::makeTBAPOMDPPrior(d, c);

        auto ba_state = p->sample(d.sampleStartState());

        WHEN("computing the probabilisty of an observation")
        {
            auto s = d.sampleStartState();
            auto o = IndexObservation(0);
            auto a = d.generateRandomAction(ba_state);

            REQUIRE(
                ba_state->computeObservationProbability(&o, a, s, rnd::sample::Dir::expectedMult)
                == 1);
            REQUIRE(
                ba_state->computeObservationProbability(&o, a, s, rnd::sample::Dir::sampleMult)
                == 1);

            d.releaseAction(a);
            d.releaseState(s);
        }

        d.releaseState(ba_state->_domain_state);
        delete (static_cast<BAPOMDPState*>(ba_state));
    }

    GIVEN("BAPOMDP State of the tiger domain with accurate counts")
    {

        c.domain_conf.domain = "episodic-tiger";
        auto const d         = domains::Tiger(domains::Tiger::TigerType::EPISODIC);
        auto const p         = factory::makeTBAPOMDPPrior(d, c);

        auto ba_state = p->sample(d.sampleStartState());

        WHEN("computing the probability of an observation")
        {
            auto listen    = IndexAction(domains::Tiger::OBSERVE),
                 open_door = IndexAction(rnd::slowRandomInt(0, 2));

            auto s = d.sampleStartState();

            auto correct_ob = IndexObservation(s->index()),
                 incorrt_ob = IndexObservation(1 - s->index());

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
        delete (static_cast<BAPOMDPState*>(ba_state));
    }
}
