#ifndef SYSADMIN_HPP
#define SYSADMIN_HPP

#include "domains/POMDP.hpp"

#include <memory>
#include <string>
#include <vector>

#include "domains/sysadmin/SysAdminParameters.hpp"
#include "domains/sysadmin/SysAdminState.hpp"
#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/Terminal.hpp"
#include "utils/DiscreteSpace.hpp"
#include "utils/distributions.hpp"
#include "utils/random.hpp"

class Reward;

namespace domains {

/**
 * @brief The sysadmin problem (either 'independent' or 'linear' version)
 *
 * A network is composed of n computers linked together by some topology. Each
 * computer is either in running or failure mode. A running computer has some
 * probability of transitioning to failure, independent of its neighbors in the
 * network; that probability is increased for every neighbor in failure mode.
 * A computer in failure mode remains so until rebooted by the operator. A
 * reward of +1 is obtained for every running computer in the network at every
 * step, no reward is given for failed computers, and a
 * -1 reward is received for each rebooting action. The goal of the operator is
 *  to maximize the number of running computers while minimizing reboots
 *  actions. The starting state assumes all computers are running.
 *
 *  The agent receives noisy signal on whether the computer it is rebooting
 *  or pinging is working (w/ the same probability)
 **/
class SysAdmin : public POMDP
{

public:
    enum NETWORK_TOPOLOGY { UNINITIALIZED, INDEPENDENT, LINEAR };
    enum Observations { FAILING, OPERATIONAL };

    static SysAdmin_Parameters const param;

    SysAdmin(int n, std::string const& version);

    /*** system parameter getters ***/
    int size() const;
    SysAdmin_Parameters const* params() const { return &param; }
    NETWORK_TOPOLOGY topology() const { return _version; }

    /**
     * @brief returns the action to observe comp
     **/
    Action const* observeAction(int comp) const;

    /**
     * @brief returns the action to reboot comp
     **/
    Action const* rebootAction(int comp) const;

    /**
     * @brief returns an updated state where computer is broken
     *
     * NOTE: does not check whether computer is failing
     **/
    State const* breakComputer(State const* s, int computer) const;

    /**
     * @brief returns an updated state where computer is fixed
     *
     * NOTE: does not check whether computer is not working
     **/
    State const* fixComputer(State const* s, int computer) const;

    /**
     * @brief returns how many failing neighbours computer c has in s
     *
     **/
    unsigned int numFailingNeighbours(int c, State const* state) const;

    /**
     * @brief returns the state associated with the input configuration
     **/
    SysAdminState const* getState(std::vector<int> config) const;

    /**
     * @brief returns the probability of computer c failing in state s after action a
     **/
    float failProbability(State const* s, Action const* a, int c) const;

    /*** environment interface ***/
    State const* sampleStartState() const final;

    Terminal step(State const** s, Action const* a, Observation const** o, Reward* r) const final;

    void releaseAction(Action const* a) const final;
    void releaseObservation(Observation const* o) const final;
    void releaseState(State const* s) const final;
    void clearCache() const final;

    Action const* copyAction(Action const* a) const final;
    Observation const* copyObservation(Observation const* o) const final;
    State const* copyState(State const* s) const final;

    /**** POMDP interface ****/
    Action const* generateRandomAction(State const* s) const final;
    void addLegalActions(State const* s, std::vector<Action const*>* actions) const final;
    double computeObservationProbability(Observation const* o, Action const* a, State const* new_s)
        const final;

private:
    NETWORK_TOPOLOGY _version;

    /**
     * @brief number of computers
     **/
    int const _size;
    int const _A_size = 2 * _size, _S_size = 0x1 << _size, _O_size = 2;
    std::vector<int> const _S_space = std::vector<int>(_size, 2);

    IndexObservation const _failing{std::to_string(FAILING)};
    IndexObservation const _operational{std::to_string(OPERATIONAL)};

    utils::DiscreteSpace<IndexAction> _actions{_A_size};
    std::vector<SysAdminState> _states;

    // mutable because generating actually modifies the distr
    mutable std::uniform_int_distribution<int> _action_distr{rnd::integerDistribution(0, _A_size)};

    /**
     * @brief returns whether this action is rebooting action
     **/
    bool isRebootingAction(Action const* a) const;

    void assertLegal(State const* s) const;
    void assertLegal(Action const* a) const;
    void assertLegal(Observation const* o) const;
};

} // namespace domains

#endif // SYSADMIN_HPP
