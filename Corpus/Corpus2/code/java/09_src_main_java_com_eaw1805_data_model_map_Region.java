package com.eaw1805.data.model.map;

import com.eaw1805.data.constants.RegionConstants;
import com.eaw1805.data.model.Game;

import java.io.Serializable;

/**
 * Represents a region of the world.
 */
public class Region implements Serializable {

    /**
     * Required by Serializable interface.
     */
    static final long serialVersionUID = 42L; //NOPMD

    /**
     * Region's identification number.
     */
    private int id; // NOPMD

    /**
     * Region's code.
     */
    private char code;

    /**
     * The name of the region.
     */
    private String name;

    /**
     * The game this region belongs to.
     */
    private Game game;

    /**
     * Default constructor.
     */
    public Region() {
        // Empty constructor
    }

    /**
     * Get the Identification number of the region.
     *
     * @return the identification number of the region.
     */
    public int getId() {
        return id;
    }

    /**
     * Set the Identification number of the region.
     *
     * @param identity the identification number of the region.
     */
    public void setId(final int identity) {
        this.id = identity;
    }

    /**
     * Get the name of the region.
     *
     * @return the name of the region.
     */
    public String getName() {
        return name;
    }

    /**
     * Set the thisName of the region.
     *
     * @param thisName the name of the region.
     */
    public void setName(final String thisName) {
        this.name = thisName;
    }

    /**
     * Get the Single-char code of the region.
     *
     * @return the Single-char code of the region.
     */
    public char getCode() {
        return code;
    }

    /**
     * Set the single-char code of the region.
     *
     * @param thisCode the single-char code of the region.
     */
    public void setCode(final char thisCode) {
        this.code = thisCode;
    }

    /**
     * Get the game this region belongs to.
     *
     * @return The game of the region.
     */
    public Game getGame() {
        return game;
    }

    /**
     * Set the game this region belongs to.
     *
     * @param value The game.
     */
    public void setGame(final Game value) {
        this.game = value;
    }

    /**
     * Indicates whether some other object is "equal to" this one.
     * The <code>equals</code> method implements an equivalence relation
     * on non-null object references:
     * <ul>
     * <li>It is <i>reflexive</i>: for any non-null reference value
     * <code>x</code>, <code>x.equals(x)</code> should return
     * <code>true</code>.
     * <li>It is <i>symmetric</i>: for any non-null reference values
     * <code>x</code> and <code>y</code>, <code>x.equals(y)</code>
     * should return <code>true</code> if and only if
     * <code>y.equals(x)</code> returns <code>true</code>.
     * <li>It is <i>transitive</i>: for any non-null reference values
     * <code>x</code>, <code>y</code>, and <code>z</code>, if
     * <code>x.equals(y)</code> returns <code>true</code> and
     * <code>y.equals(z)</code> returns <code>true</code>, then
     * <code>x.equals(z)</code> should return <code>true</code>.
     * <li>It is <i>consistent</i>: for any non-null reference values
     * <code>x</code> and <code>y</code>, multiple invocations of
     * <tt>x.equals(y)</tt> consistently return <code>true</code>
     * or consistently return <code>false</code>, provided no
     * information used in <code>equals</code> comparisons on the
     * objects is modified.
     * <li>For any non-null reference value <code>x</code>,
     * <code>x.equals(null)</code> should return <code>false</code>.
     * </ul>
     * The <tt>equals</tt> method for class <code>Object</code> implements
     * the most discriminating possible equivalence relation on objects;
     * that is, for any non-null reference values <code>x</code> and
     * <code>y</code>, this method returns <code>true</code> if and only
     * if <code>x</code> and <code>y</code> refer to the same object
     * (<code>x == y</code> has the value <code>true</code>).
     * Note that it is generally necessary to override the <tt>hashCode</tt>
     * method whenever this method is overridden, so as to maintain the
     * general contract for the <tt>hashCode</tt> method, which states
     * that equal objects must have equal hash codes.
     *
     * @param obj the reference object with which to compare.
     * @return <code>true</code> if this object is the same as the obj
     * argument; <code>false</code> otherwise.
     * @see #hashCode()
     * @see java.util.Hashtable
     */
    @Override
    public boolean equals(final Object obj) {
        if (obj == null) {
            return false;
        }

        if (!(obj instanceof Region)) {
            return false;
        }

        final Region region = (Region) obj;

        if (code != region.code) {
            return false;
        }
        if (id != region.id) {
            return false;
        }
        if (name != null ? !name.equals(region.name) : region.name != null) {
            return false;
        }

        return true;
    }

    /**
     * Returns a hash code value for the object. This method is
     * supported for the benefit of hashtables such as those provided by
     * <code>java.util.Hashtable</code>.
     * The general contract of <code>hashCode</code> is:
     * <ul>
     * <li>Whenever it is invoked on the same object more than once during
     * an execution of a Java application, the <tt>hashCode</tt> method
     * must consistently return the same integer, provided no information
     * used in <tt>equals</tt> comparisons on the object is modified.
     * This integer need not remain consistent from one execution of an
     * application to another execution of the same application.
     * <li>If two objects are equal according to the <tt>equals(Object)</tt>
     * method, then calling the <code>hashCode</code> method on each of
     * the two objects must produce the same integer result.
     * <li>It is <em>not</em> required that if two objects are unequal
     * according to the {@link java.lang.Object#equals(java.lang.Object)}
     * method, then calling the <tt>hashCode</tt> method on each of the
     * two objects must produce distinct integer results.  However, the
     * programmer should be aware that producing distinct integer results
     * for unequal objects may improve the performance of hashtables.
     * </ul>
     * As much as is reasonably practical, the hashCode method defined by
     * class <tt>Object</tt> does return distinct integers for distinct
     * objects. (This is typically implemented by converting the internal
     * address of the object into an integer, but this implementation
     * technique is not required by the
     * Java<font size="-2"><sup>TM</sup></font> programming language.)
     *
     * @return a hash code value for this object.
     * @see java.lang.Object#equals(java.lang.Object)
     * @see java.util.Hashtable
     */
    @Override
    public int hashCode() {
        return id;
    }

    @Override
    public String toString() {
        final StringBuilder sbld = new StringBuilder();
        switch (id) {
            case RegionConstants.EUROPE:
                sbld.append("E");
                break;

            case RegionConstants.CARIBBEAN:
                sbld.append("C");
                break;

            case RegionConstants.INDIES:
                sbld.append("I");
                break;

            case RegionConstants.AFRICA:
                sbld.append("A");
                break;

            default:
                break;

        }

        return sbld.toString();
    }

}
