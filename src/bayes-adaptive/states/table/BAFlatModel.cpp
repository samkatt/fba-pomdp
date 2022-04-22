#include "BAFlatModel.hpp"

#include <algorithm> // std::copy
#include <cassert>

#include "easylogging++.h"

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"

#include "utils/index.hpp"

namespace bayes_adaptive { namespace table {

BAFlatModel::BAFlatModel() : _domain_size(0) {}

BAFlatModel::BAFlatModel(Domain_Size const* domain_size) :
        _domain_size(domain_size),
        _phi(std::make_shared<std::vector<float>>(
            domain_size->_S * domain_size->_A * domain_size->_S)),
        _psi(std::make_shared<std::vector<float>>(
            domain_size->_A * domain_size->_S * domain_size->_O))
{
    assert(
        _phi->size()
        == static_cast<size_t>(_domain_size->_S * _domain_size->_A * _domain_size->_S));
    assert(
        _psi->size()
        == static_cast<size_t>(_domain_size->_A * _domain_size->_S * _domain_size->_O));
}

BAFlatModel::BAFlatModel(
    std::shared_ptr<std::vector<float> const> phi,
    std::shared_ptr<std::vector<float> const> psi,
    Domain_Size const* domain_size) :
        _domain_size(domain_size), _phi(std::move(phi)), _psi(std::move(psi))
{
    assert(
        _phi->size()
        == static_cast<size_t>(_domain_size->_S * _domain_size->_A * _domain_size->_S));
    assert(
        _psi->size()
        == static_cast<size_t>(_domain_size->_A * _domain_size->_S * _domain_size->_O));
}

float& BAFlatModel::count(State const* s, Action const* a, State const* new_s)
{

    assertLegal(s);
    assertLegal(a);
    assertLegal(new_s);

    return phi(s->index(), a->index(), new_s->index());
}

float& BAFlatModel::count(Action const* a, State const* new_s, Observation const* o)
{

    assertLegal(o);
    assertLegal(a);
    assertLegal(new_s);

    return psi(a->index(), new_s->index(), o->index());
}

std::vector<float> BAFlatModel::transitionExpectation(State const* s, Action const* a) const
{
    assertLegal(s);
    assertLegal(a);

    return rnd::sample::Dir::expectedMult(&phi(s->index(), a->index(), 0), _domain_size->_S);
}

std::vector<float> BAFlatModel::observationExpectation(Action const* a, State const* new_s) const
{
    assertLegal(a);
    assertLegal(new_s);

    return rnd::sample::Dir::expectedMult(&psi(a->index(), new_s->index(), 0), _domain_size->_O);
}

int BAFlatModel::sampleStateIndex(State const* s, Action const* a, rnd::sample::Dir::sampleMethod m)
    const
{

    assertLegal(s);
    assertLegal(a);

    return m(&phi(s->index(), a->index(), 0), _domain_size->_S);
}

int BAFlatModel::sampleObservationIndex(
    Action const* a,
    State const* new_s,
    rnd::sample::Dir::sampleMethod m) const
{

    assertLegal(a);
    assertLegal(new_s);

    return m(&psi(a->index(), new_s->index(), 0), _domain_size->_O);
}

double BAFlatModel::computeObservationProbability(
    Observation const* o,
    Action const* a,
    State const* new_s,
    rnd::sample::Dir::sampleMultinominal m) const
{

    assertLegal(o);
    assertLegal(a);
    assertLegal(new_s);

    // corner (easy) case
    if (_domain_size->_O == 1)
    {
        return 1;
    }

    // samples multinominal & returns the correct index
    return m(&psi(a->index(), new_s->index(), 0), _domain_size->_O)[o->index()];
}

void BAFlatModel::incrementCountsOf(
    State const* s,
    Action const* a,
    Observation const* o,
    State const* new_s,
    float amount)
{

    assertLegal(s);
    assertLegal(a);
    assertLegal(o);
    assertLegal(new_s);

    count(s, a, new_s) += amount;
    count(a, new_s, o) += amount;
}

void BAFlatModel::logCounts() const
{

    LOG(INFO) << "Counts for BAPOMDP state:";
    LOG(INFO) << "Transition:";

    for (auto s = 0; s < _domain_size->_S; ++s)
    {
        for (auto a = 0; a < _domain_size->_A; ++a)
        {
            std::string dirichlet_descr = "{" + std::to_string(phi(s, a, 0));

            for (auto new_s = 1; new_s < _domain_size->_S; ++new_s)
            {
                dirichlet_descr += ", " + std::to_string(phi(s, a, new_s));
            }

            dirichlet_descr += "}";

            LOG(INFO) << "<s:" << s << ", a:" << a << ">: " << dirichlet_descr;
        }
    }

    LOG(INFO) << "Observation:";
    for (auto a = 0; a < _domain_size->_A; ++a)
    {
        for (auto s = 0; s < _domain_size->_S; ++s)
        {
            std::string dirichlet_descr = "{" + std::to_string(psi(a, s, 0));

            for (auto o = 1; o < _domain_size->_O; ++o)
            {
                dirichlet_descr += ", " + std::to_string(psi(a, s, o));
            }

            dirichlet_descr += "}";

            LOG(INFO) << "<a:" << a << ", s:" << s << ">: " << dirichlet_descr;
        }
    }
}

float& BAFlatModel::phi(int s, int a, int new_s)
{

    auto const delta_index = phi_cache_index(s, a);
    auto cached_phi        = _phi_cache.find(delta_index);

    // store relevant dir if not already present
    if (cached_phi == _phi_cache.end())
    {

        auto const phi_index = indexing::threeToOne(s, a, 0, _domain_size->_A, _domain_size->_S);

        cached_phi = _phi_cache
                         .insert(
                             {delta_index,
                              std::vector<float>(
                                  &_phi->at(phi_index), &_phi->at(phi_index) + _domain_size->_S)})
                         .first;
    }

    return cached_phi->second[new_s];
}

float const& BAFlatModel::phi(int s, int a, int new_s) const
{

    auto const delta_index = phi_cache_index(s, a);
    auto const delta_val   = _phi_cache.find(delta_index);

    return (delta_val != _phi_cache.end())
               ? delta_val->second[new_s]
               : _phi->at(indexing::threeToOne(s, a, new_s, _domain_size->_A, _domain_size->_S));
}

float& BAFlatModel::psi(int a, int new_s, int o)
{

    auto const delta_index = psi_cache_index(a, new_s);
    auto cached_psi        = _psi_cache.find(delta_index);

    // store relevant dir if not already present
    if (cached_psi == _psi_cache.end())
    {

        auto const psi_index =
            indexing::threeToOne(a, new_s, 0, _domain_size->_S, _domain_size->_O);

        cached_psi = _psi_cache
                         .insert(
                             {delta_index,
                              std::vector<float>(
                                  &_psi->at(psi_index), &_psi->at(psi_index) + _domain_size->_O)})
                         .first;
    }

    return cached_psi->second[o];
}

float const& BAFlatModel::psi(int a, int new_s, int o) const
{

    auto const delta_index = psi_cache_index(a, new_s);
    auto const delta_val   = _psi_cache.find(delta_index);

    return (delta_val != _psi_cache.end())
               ? delta_val->second[o]
               : _psi->at(indexing::threeToOne(a, new_s, o, _domain_size->_S, _domain_size->_O));
}

unsigned int BAFlatModel::phi_cache_index(int s, int a) const
{
    return s * _domain_size->_A + a;
}

unsigned int BAFlatModel::psi_cache_index(int a, int new_s) const
{
    return a * _domain_size->_S + new_s;
}

void BAFlatModel::assertLegal(State const* s) const
{
    assert(s && s->index() >= 0 && s->index() < _domain_size->_S);
}

void BAFlatModel::assertLegal(Action const* a) const
{
    assert(a && a->index() >= 0 && a->index() < _domain_size->_A);
}

void BAFlatModel::assertLegal(Observation const* o) const
{
    assert(o && o->index() >= 0 && o->index() < _domain_size->_O);
}

BAFlatModel::BAFlatModel(BAFlatModel const& other) : _domain_size()
{
    *this = other;
}

BAFlatModel& BAFlatModel::operator=(BAFlatModel const& other)
{

    _domain_size = other._domain_size;

    // merge if necessary
    if (other._phi_cache.size() < _cache_ratio_threshold * _domain_size->_A * _domain_size->_S)
    {

        // cache still small enough
        _phi       = other._phi;
        _phi_cache = other._phi_cache;

    } else // cache too large
    {

        _phi_cache = {};

        // base case is a copy of other phi
        auto phi = *other._phi;

        // update all dir in phi_cache
        for (auto const& it : other._phi_cache)
        {
            auto const s = it.first / _domain_size->_A;
            auto const a = it.first % _domain_size->_A;

            auto phi_index = indexing::threeToOne(s, a, 0, _domain_size->_A, _domain_size->_S);

            // update all entries in dir
            std::copy(it.second.begin(), it.second.end(), &phi[phi_index]);
        }

        // store our new phi as shared pointer
        _phi = std::make_shared<std::vector<float>>(std::move(phi));
    }

    // merge if necessary
    if (other._psi_cache.size() < _cache_ratio_threshold * _domain_size->_A * _domain_size->_S)
    {

        // cache still small enough
        _psi       = other._psi;
        _psi_cache = other._psi_cache;

    } else // cache too large
    {

        _psi_cache = {};

        // base case is a copy of other psi
        auto psi = *other._psi;

        // update all dir in psi_cache
        for (auto const& it : other._psi_cache)
        {
            auto const a     = it.first / _domain_size->_S;
            auto const new_s = it.first % _domain_size->_S;

            auto psi_index = indexing::threeToOne(a, new_s, 0, _domain_size->_S, _domain_size->_O);

            // update all entries in dir
            std::copy(it.second.begin(), it.second.end(), &psi[psi_index]);
        }

        // store our new psi as shared pointer
        _psi = std::make_shared<std::vector<float>>(std::move(psi));
    }

    return *this;
}

}} // namespace bayes_adaptive::table
