#include "GridWorldButtons.hpp"

#include <algorithm> // Why?
#include <cstdlib>

#include "easylogging++.h"

#include "environment/Reward.hpp"
#include "utils/index.hpp"
#include "utils/random.hpp"

namespace domains {

std::vector<std::string> const GridWorldButtons::GridWorldButtonsAction::action_descriptions = {"UP",
                                                                                  "RIGHT",
                                                                                  "DOWN",
                                                                                  "LEFT",
                                                                                  "PRESS",
                                                                                  "HIT"};

GridWorldButtons::GridWorldButtonsState::pos const GridWorldButtons::start_location = {0, 0};
std::vector<GridWorldButtons::GridWorldButtonsState::pos> const GridWorldButtons::goal_locations = {{0,5},{1,5},{2,5}};
std::vector<GridWorldButtons::GridWorldButtonsState::pos> const GridWorldButtons::wall_tiles = {};
// {0,4},{0,5},{0,6},{1,3},{1,4},{1,5},{1,6},{3,1},{3,2},{3,3},{3,4}, {3,5}};


GridWorldButtons::GridWorldButtons(size_t extra_features ):
    _extra_features(extra_features),
    _S_size(0), // initiated below
    _O_size(0) // initiated below
{
    if (_extra_features > 20) {
        throw "please enter a size between 0 and 15 to be able to run gridworldButtons \
                    (you entered " + std::to_string(_extra_features) + ")";
    }
    // Number of states, widthxheight grid minus some walls, correct button, door open/closed, door_hp X, extra features binary.
    _S_size = _size_width * _size_height * 3 * 2 * (door_max_hp + 1) * std::pow(2, _extra_features); // 12 blocks the agent can not enter
    // Position, d1 open/closed, d2 open/closed
    // for the position, we have 11 observations. The observation function is deterministic, but on some locations the
    // obserations won't allow to distinguish between the rel positions. (e.g. in the hallway you won't exactly know your position)
    _O_size = 6 * 2; //

    _S.reserve(_S_size);
    _O.reserve(_O_size);

    std::vector<int> passed_observation_indexes;
    // generate state space
    for (unsigned int x_agent = 0; x_agent < _size_width; ++x_agent) {
        for (unsigned int y_agent = 0; y_agent < _size_height; ++y_agent) {
            GridWorldButtonsState::pos const agent_pos{x_agent, y_agent};
//            if (! std::count(wall_tiles.begin(), wall_tiles.end(), agent_pos)) {
            for (unsigned int door_open = 0; door_open < 2; ++door_open) {
                for (unsigned int correct_button = 0; correct_button < 3; ++correct_button) {
                    for (unsigned int door_hp = 0; door_hp < (door_max_hp + 1); ++door_hp) {
                        for (unsigned int feature_config = 0;
                             feature_config < std::pow(2, extra_features); ++feature_config) {

                            auto const i = positionsToIndex(agent_pos, correct_button, door_open, door_hp, feature_config);

                            assert(static_cast<unsigned int>(i) == _S.size());
                            auto const state = GridWorldButtonsState(agent_pos, correct_button, door_open, door_hp, feature_config, i);
                            assertLegal(&state);

                            _S.emplace_back(state);
                        }
                    }
                }
                auto const j = positionsToObservationIndex(agent_pos, door_open);

                if (! std::count(passed_observation_indexes.begin(), passed_observation_indexes.end(), j)) {
                    passed_observation_indexes.emplace_back(j);
                    assert(static_cast<unsigned int>(j) == _O.size());
                    _O.emplace_back(GridWorldButtonsObservation(agent_pos, door_open, j));
                }
            }
//            }
        }
    }

    VLOG(1) << "initiated gridworld-buttons";
}

bool isWallTile(GridWorldButtons::GridWorldButtonsState::pos const& position) {
    if (std::count(GridWorldButtons::wall_tiles.begin(), GridWorldButtons::wall_tiles.end(), position)) {
        return true;
    }
    return false;
}

size_t GridWorldButtons::size_width() const
{
    return _size_width;
}

size_t GridWorldButtons::size_height() const
{
    return _size_height;
}

bool GridWorldButtons::foundGoal(GridWorldButtonsState const* s) const
{
    if (std::count(goal_locations.begin(), goal_locations.end(), s->_agent_position)) {
        return true;
    }
    return false;
}

GridWorldButtons::GridWorldButtonsState const* GridWorldButtons::getState(
        GridWorldButtonsState::pos const& agent_pos, unsigned int const& correct_button,
        unsigned int const& d_open, unsigned int const& d_hp, unsigned int const& feature_config) const
{
    return &_S[positionsToIndex(agent_pos, correct_button, d_open, d_hp, feature_config)];
}

unsigned int GridWorldButtons::locationToObservationLocation(GridWorldButtons::GridWorldButtonsState::pos agent_pos) {
    if (agent_pos.x == 0 && agent_pos.y == 3) {
        return 1;
    } else if ((agent_pos.x == 0 || agent_pos.x == 1) && agent_pos.y == 4) {
        return 2;
    } else if (std::count(goal_locations.begin(), goal_locations.end(), agent_pos)){
        return 3;
    } else if (agent_pos.x == 1 && agent_pos.y == 2) {
        return 4;
    } else if (agent_pos.x == 2 && agent_pos.y == 4) {
        return 5;
    } else {
        return 0;
    }
}


GridWorldButtons::GridWorldButtonsObservation const* GridWorldButtons::getObservation(
    GridWorldButtonsState::pos const& agent_pos, unsigned int const& d_open) const
{
    return &_O[positionsToObservationIndex(agent_pos, d_open)];
}

Action const* GridWorldButtons::generateRandomAction(State const* s) const
{
    assertLegal(s);
    return new GridWorldButtonsAction(rnd::slowRandomInt(0, _A_size));
}

void GridWorldButtons::addLegalActions(State const* s, std::vector<Action const*>* actions) const
{
    assert(actions != nullptr && actions->empty());
    assertLegal(s);

    for (auto a = 0; a < _A_size; ++a) { actions->emplace_back(new GridWorldButtonsAction(a)); }
}

double GridWorldButtons::computeObservationProbability(
    Observation const* o,
    Action const* a,
    State const* new_s) const
{
    assertLegal(o);
    assertLegal(a);
    assertLegal(new_s);

    if (static_cast<GridWorldButtonsObservation const*>(o)->_location_observation ==
        locationToObservationLocation(static_cast<GridWorldButtonsState const*>(new_s)->_agent_position) &&
            static_cast<GridWorldButtonsState const*>(new_s)->_d_open == static_cast<GridWorldButtonsObservation const*>(o)->_d_open) {
        return 1;
    }
    return 0;

}

void GridWorldButtons::releaseAction(Action const* a) const
{
    assertLegal(a);
    if (a != nullptr) {
        delete static_cast<GridWorldButtonsAction*>(const_cast<Action*>(a));
    }
}

Action const* GridWorldButtons::copyAction(Action const* a) const
{
    assertLegal(a);
    return new GridWorldButtonsAction(std::stoi(a->index()));
}

State const* GridWorldButtons::sampleStartState() const
{
    // sample random start state, fixed location but random correct button and feature configuration.
    auto const agent_pos = start_location;
    auto correct_button = rnd::slowRandomInt(0,3);
    auto feature_config = rnd::slowRandomInt(0, std::pow(2,_extra_features));
    auto index = positionsToIndex(agent_pos, correct_button, 0, door_max_hp, feature_config);
    return &_S[index];
}

Terminal GridWorldButtons::step(State const** s, Action const* a, Observation const** o, Reward* r) const
{
    assert(s != nullptr && o != nullptr && r != nullptr);
    assertLegal(*s);
    assertLegal(a);

    auto grid_state       = static_cast<GridWorldButtonsState const*>(*s);
    auto const& agent_pos = grid_state->_agent_position;
    auto new_d_hp = grid_state->_d_hp;
    auto new_d_open = grid_state->_d_open;

//    int button_pressed = 0;
    int button_opened_door = 0;
    /*** T ***/
    switch (std::stoi(a->index()))
    {
        case GridWorldButtonsAction::ACTION::PRESS:  // button press
            if ((agent_pos.x == 0 && agent_pos.y == 3) || (agent_pos.x == 1 && agent_pos.y == 2)
                || (agent_pos.x == 2 && agent_pos.y == 4)) {
//                button_pressed++;
            }
            if (agent_pos.x == 0 && agent_pos.y == 3 && grid_state->_correct_button == 0) {
                if (rnd::uniform_rand01() < button_A_open_prob) {
                    new_d_open = 1;
                    if (new_d_open && !grid_state->_d_open){
                        button_opened_door++;
                    }
                }
            } else if (agent_pos.x == 1 && agent_pos.y == 2 && grid_state->_correct_button == 1) {
                if (rnd::uniform_rand01() < button_B_open_prob) {
                    new_d_open = 1;
                    if (new_d_open && !grid_state->_d_open){
                        button_opened_door++;
                    }
                }
            } else if (agent_pos.x == 2 && agent_pos.y == 4 && grid_state->_correct_button == 2) {
                if (rnd::uniform_rand01() < button_C_open_prob) {
                    new_d_open = 1;
                    if (new_d_open && !grid_state->_d_open){
                        button_opened_door++;
                    }
                }
            }
            break;
        case GridWorldButtonsAction::ACTION::HIT:  // hit door
            if ((agent_pos.x == 0 && agent_pos.y == 4)
                || (agent_pos.x == 1 && agent_pos.y == 4)
                || (agent_pos.x == 2 && agent_pos.y == 4)) {
                if (new_d_hp > 0) {
                    new_d_hp--;
                    if (new_d_hp == 0) {
                        new_d_open = 1;
                    }
                }
            }
            break;
    }



    // calculate whether move succeeds
    bool const move_succeeds = rnd::uniform_rand01() < move_prob;

    auto new_agent_pos = (move_succeeds) ? applyMove(agent_pos, a, grid_state->_d_open) : agent_pos;

    auto const found_goal = foundGoal(grid_state);

    auto const new_index = positionsToIndex(new_agent_pos, grid_state->_correct_button, new_d_open, new_d_hp, grid_state->_feature_config);
    *s                   = &_S[new_index];

    /*** R & O ***/
    r->set((found_goal ?
     (grid_state->_d_hp > 0 ? goal_reward_big : goal_reward_small)
    : step_reward) + button_opened_door*button_press_reward);
    *o = getObservation(new_agent_pos, new_d_open);

    return Terminal(found_goal);
}

void GridWorldButtons::releaseObservation(Observation const* o) const
{
    assertLegal(o);
}

void GridWorldButtons::releaseState(State const* s) const
{
    assertLegal(s);
}

Observation const* GridWorldButtons::copyObservation(Observation const* o) const
{
    assertLegal(o);
    return o;
}

State const* GridWorldButtons::copyState(State const* s) const
{
    assertLegal(s);
    return s;
}

int GridWorldButtons::positionsToIndex(
    GridWorldButtonsState::pos const& agent_pos,unsigned int const& correct_button,
    unsigned int const& d_open, unsigned int const& d_hp, unsigned int const& feature_config) const
{
    // indexing: from 5 elements projecting to 1 dimension
    // x*height*3*2*d2_max_hp * pow(2,_extra_features) ... + feature_config
    return agent_pos.x * _size_height * 2 * 3 * (door_max_hp + 1) * std::pow(2, _extra_features)
    + agent_pos.y * 3 * 2 * (door_max_hp + 1) * std::pow(2, _extra_features)
    + d_open * 3 * (door_max_hp + 1) * std::pow(2, _extra_features)
    + correct_button * (door_max_hp + 1) * std::pow(2, _extra_features)
    + d_hp * std::pow(2, _extra_features)
    + feature_config;
}

int GridWorldButtons::positionsToObservationIndex(GridWorldButtonsState::pos const& agent_pos,
                                                  const unsigned  int& d_open) const
{
    // indexing: from 3 elements projecting to 1 dimension
    // obs*2 + d1_open
    return locationToObservationLocation(agent_pos) * 2 + d_open;
}


GridWorldButtons::GridWorldButtonsState::pos
GridWorldButtons::applyMove(GridWorldButtonsState::pos const& old_pos, Action const* a, unsigned int const& d_open) const
{
    assertLegal(a);

    auto x = old_pos.x;
    auto y = old_pos.y;

    // move according to action
    switch (std::stoi(a->index()))
    {
        case GridWorldButtonsAction::ACTION::UP:
            y = (old_pos.y == _size_height - 1) ? old_pos.y :
                     (!d_open && old_pos.y == 4 ? old_pos.y : old_pos.y + 1);
            break;
        case GridWorldButtonsAction::ACTION::DOWN:
            y = (old_pos.y == 0) ? old_pos.y :
                   (isWallTile({old_pos.x,old_pos.y-1}) ? old_pos.y : old_pos.y - 1);
            break;
        case GridWorldButtonsAction::ACTION::RIGHT:
            x = ((old_pos.x == _size_width - 1) || (old_pos.y > 0 && old_pos.y < 6)) ? old_pos.x :
                (isWallTile({old_pos.x + 1,old_pos.y}) ? old_pos.x :
                 (!d_open && old_pos.x == 0 && old_pos.y == 6 ? old_pos.x : old_pos.x + 1));
            break;
        case GridWorldButtonsAction::ACTION::LEFT: x =
                ((old_pos.x == 0) || (old_pos.y > 0 && old_pos.y < 6)) ? old_pos.x :
           (isWallTile({old_pos.x - 1,old_pos.y}) ? old_pos.x :
            (!d_open && old_pos.x == 4 && old_pos.y == 6 ? old_pos.x : old_pos.x - 1));
           break;
    }
    if (old_pos.y == 4 && !d_open && y == 5) {
        y = 5;
    }

    return {x, y};
}

void GridWorldButtons::assertLegal(Action const* a) const
{
    assert(a != nullptr);
    assert(std::stoi(a->index()) >= 0 && std::stoi(a->index()) < _A_size);
}

void GridWorldButtons::assertLegal(Observation const* o) const
{
    assert(o != nullptr);
    assert(std::stoi(o->index()) >= 0 && std::stoi(o->index()) < _O_size);
    assert(static_cast<GridWorldButtonsObservation const*>(o)->_location_observation < 15);
    assert(static_cast<GridWorldButtonsObservation const*>(o)->_d_open <= 1);
}

void GridWorldButtons::assertLegal(State const* s) const
{
    assert(s != nullptr);
    assert(std::stoi(s->index()) >= 0 &&std::stoi(s->index())< _S_size);
    assertLegal(static_cast<GridWorldButtonsState const*>(s)->_agent_position);
}

void GridWorldButtons::assertLegal(GridWorldButtonsState::pos const& position) const
{
    assert(position.x < _size_width);
    assert(position.y < _size_height);
}

} // namespace domains
