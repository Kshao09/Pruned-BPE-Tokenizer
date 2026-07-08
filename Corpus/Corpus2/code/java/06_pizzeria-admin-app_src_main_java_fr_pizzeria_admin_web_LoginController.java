package fr.pizzeria.admin.web;

import java.io.IOException;

import javax.servlet.RequestDispatcher;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import org.apache.commons.lang3.StringUtils;

@WebServlet("/login")
public class LoginController extends HttpServlet {

	


	@Override
	protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
		RequestDispatcher rd = this.getServletContext().getRequestDispatcher("/WEB-INF/views/login/login.jsp");
		rd.forward(req, resp);
	}
	
	
	@Override
	protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
		String email = req.getParameter("email");
		String password = req.getParameter("password");
		
		if (StringUtils.isBlank(email) || StringUtils.isBlank(password)) 
		{
			resp.sendError(400, "Non non non ! Zone interdite");
		} 
		else if ( StringUtils.equals(email, "admin@pizzeria.fr") 
			&&  StringUtils.equals(password, "admin"))
		{
			HttpSession session = req.getSession();
			session.setAttribute("email", email);
			resp.sendRedirect(req.getContextPath() + "/pizzas/list");
		} 
		else
		{
			resp.setStatus(403);
			req.setAttribute("msgerr", "Ooppps   noooo");
			RequestDispatcher rd = this.getServletContext().getRequestDispatcher("/WEB-INF/views/login/login.jsp");
			rd.forward(req, resp);
		
		}
	}

	

	
}
