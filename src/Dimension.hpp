#ifndef DIMENSION_HPP
#define DIMENSION_HPP

class Dim1
{
public:
    int x;

public:
    Dim1(): x(0)
    { }

    explicit Dim1(int x_): x(x_)
    { }

    Dim1 operator+(const Dim1& rhs) const
    {
        return Dim1(x + rhs.x);
    }

    Dim1 operator-(const Dim1& rhs) const
    {
        return Dim1(x - rhs.x);
    }

    Dim1 operator*(const Dim1& rhs) const
    {
        return Dim1(x * rhs.x);
    }
};

class Dim2
{
public:
    int x;
    int y;

public:
    Dim2(): x(0), y(0)
    { }

    Dim2(int x_, int y_): x(x_), y(y_)
    { }

    int Product() const
    {
        return x * y;
    }

    int Area() const
    {
        return Product();
    }

    void Wrap(int width)
    {
        y += x / width;
        x %= width;
    }

    /// Increments x by one, but wraps x at width, at which point, it resets
    /// x and increments y by one.
    void Inc(int width)
    {
        ++x;
        Wrap(width);
    }

    /// Component-wise addition
    Dim2 operator+(const Dim2& rhs) const
    {
        return Dim2(x + rhs.x, y + rhs.y);
    }

    /// Component-wise subtraction
    Dim2 operator-(const Dim2& rhs) const
    {
        return Dim2(x - rhs.x, y - rhs.y);
    }

    /// Component-wise multiplication
    Dim2 operator*(const Dim2& rhs) const
    {
        return Dim2(x * rhs.x, y * rhs.y);
    }
};

#endif // DIMENSION_HPP
