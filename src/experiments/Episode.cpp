#include "Episode.hpp"

#include "beliefs/Belief.hpp"
#include "domains/POMDP.hpp"
#include "environment/Action.hpp"
#include "environment/Environment.hpp"
#include "environment/History.hpp"
#include "environment/Horizon.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"
#include "environment/Terminal.hpp"
#include "planners/Planner.hpp"

namespace episode {
Result
    run(Planner const& planner,
        Belief& belief,
        Environment const& env,
        POMDP const& simulator,
        Horizon const& h,
        Discount discount)
{
    assert(h.toInt() > 0);
    assert(discount.toDouble() >= 0 && discount.toDouble() <= 1);

    auto r        = Reward(0);
    auto ret      = Return();
    auto terminal = Terminal(false);
    Observation const* o;

    State const* s = env.sampleStartState();
    History hist;

    VLOG(2) << "Episode starts with s=" << s->toString();

    // interact until horizon or terminal interaction occurred
    int t;
    for (t = 0; t < h.toInt() && !terminal.terminated(); ++t)
    {
        auto const a = planner.selectAction(simulator, belief, hist);
        terminal     = env.step(&s, a, &o, &r);

        VLOG(2) << "T=" << t << "\ta=" << a->toString() << "\ts'=" << s->toString()
                << "\to=" << o->toString() << "\tr=" << r.toDouble();

        if (!terminal.terminated())
        {
            belief.updateEstimation(a, o, simulator);
        }

        ret.add(r, discount);
        discount.increment();

        hist.add(a, o);
    }

    VLOG(2) << "End of episode at s=" << s->toString() << " with return=" << ret.toDouble();

    hist.clear(simulator, env);
    env.releaseState(s);

    return {ret, t};
}

} // namespace episode
