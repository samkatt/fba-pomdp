#include "catch.hpp"

#include <memory>
#include <utility>

#include "bayes-adaptive/models/table/BADomainExtension.hpp"
#include "bayes-adaptive/models/table/BAPOMDP.hpp"
#include "bayes-adaptive/priors/BAPOMDPPrior.hpp"
#include "domains/dummy/DummyDomain.hpp"
#include "domains/dummy/DummyDomainBAExtension.hpp"

#include "configurations/BAConf.hpp"

#include "bayes-adaptive/states/table/BAPOMDPState.hpp"
#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"
#include "environment/Terminal.hpp"

#include "utils/random.hpp"

TEST_CASE("step function", "[bayes-adaptive]")
{
    GIVEN("The bayes-adaptive dummy domain")
    {
        configurations::BAConf c;

        for (auto m = 0; m < 2; ++m)
        {
            WHEN("creating a new state and stepping with sample method " + std::to_string(m))
            {

                c.domain_conf.domain = "dummy";

                auto domain = std::unique_ptr<POMDP>(new domains::DummyDomain());
                auto ext    = std::unique_ptr<BADomainExtension>(
                    new bayes_adaptive::domain_extensions::DummyDomainBAExtension());
                auto prior = factory::makeTBAPOMDPPrior(*domain, c);

                auto sample_method = (m == 0) ? rnd::sample::Dir::sampleFromSampledMult
                                              : rnd::sample::Dir::sampleFromExpectedMult;

                auto compute_mult_method =
                    (m == 0) ? rnd::sample::Dir::sampleMult : rnd::sample::Dir::expectedMult;

                auto update_abstract_model = false;

                BAPOMDP d(
                    std::move(domain),
                    std::move(ext),
                    std::move(prior),
                    sample_method,
                    compute_mult_method,
                 update_abstract_model);

                auto s               = d.sampleStartState();
                auto a               = d.generateRandomAction(s);
                Observation const* o = nullptr;
                Reward r(0);

                auto ba_new_s = const_cast<BAPOMDPState*>(static_cast<BAPOMDPState const*>(s));
                auto ba_old_s = const_cast<BAPOMDPState*>(
                    static_cast<BAPOMDPState const*>(d.copyState(ba_new_s)));

                THEN("counts should increase")
                {
                    d.step(&s, a, &o, &r);

                    auto c_s = ba_old_s->model()->count(s, a, s);
                    auto o_s = ba_old_s->model()->count(a, s, o);

                    auto new_c_s = ba_new_s->model()->count(s, a, s);
                    auto new_o_s = ba_new_s->model()->count(a, s, o);

                    REQUIRE(c_s == new_c_s - 1.);
                    REQUIRE(o_s == new_o_s - 1.);
                }

                AND_WHEN("we use mode KeepCounts")
                {
                    d.mode(BAPOMDP::StepType::KeepCounts);
                    d.step(&s, a, &o, &r);

                    THEN("counts should stay constant")
                    {
                        auto c_s = ba_old_s->model()->count(s, a, s);
                        auto o_s = ba_old_s->model()->count(a, s, o);

                        auto new_c_s = ba_new_s->model()->count(s, a, s);
                        auto new_o_s = ba_new_s->model()->count(a, s, o);

                        REQUIRE(c_s == new_c_s);
                        REQUIRE(o_s == new_o_s);
                    }
                }

                THEN("new state & observation should be 0")
                {
                    d.step(&s, a, &o, &r);

                    REQUIRE(s->index() == 0);
                    REQUIRE(std::stoi(a->index()) == 0);
                    REQUIRE(std::stoi(o->index()) == 0);
                }

                d.releaseAction(a);
                d.releaseState(ba_old_s);
                d.releaseState(s);
                d.releaseObservation(o);
            }
        }
    }
}
