package org.anodyneos.xp.tag.core;

import javax.servlet.jsp.el.ELException;

import org.anodyneos.xp.XpException;
import org.anodyneos.xp.XpOutput;
import org.anodyneos.xp.tagext.XpTagSupport;
import org.xml.sax.SAXException;

/**
 * @author jvas
 */
public class DebugTag extends XpTagSupport {

    public DebugTag() {
        super();
    }

    /* (non-Javadoc)
     * @see org.anodyneos.xp.tagext.XpTag#doTag(org.anodyneos.xp.XpContentHandler)
     */
    public void doTag(XpOutput out) throws XpException, ELException, SAXException {
        XpOutput newOut = new XpOutput(new DebugCH(System.err, out.getXpContentHandler()));
        getXpBody().invoke(newOut);
    }

}
