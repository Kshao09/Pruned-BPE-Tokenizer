#ifndef IDUCK_HPP_124b1b04_74d3_41ae_b7ce_2a0bea79f1c7
#define IDUCK_HPP_124b1b04_74d3_41ae_b7ce_2a0bea79f1c7

#include "../strategy/IFly.hpp"

/**
  \interface IDuck
  Interface for duck. It can fly, quack and rotate right.
  */

class IDuck
{
public:
    virtual Course getCourse() const = 0;
    virtual int getDistance(Course course) const = 0;
    virtual void fly() = 0;
    virtual void quack() = 0;
    virtual void right() = 0;
    virtual void left() = 0;

    virtual ~IDuck() { }
};

#endif // IDUCK_HPP_124b1b04_74d3_41ae_b7ce_2a0bea79f1c7
