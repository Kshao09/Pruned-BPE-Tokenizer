angular.module('appTesting').service("LoginLocalStorage", function () {
    "use strict";

    var STORE_NAME = "login";

    var setUser = function setUser(user) {
        localStorage.setItem(STORE_NAME, JSON.stringify(user));
    }

    var getUser = function getUser() {
        var storedTasks = localStorage.getItem(STORE_NAME);
        if (storedTasks) {
            return JSON.parse(storedTasks);
        }
        return {};
    }

    return {
        setUser: setUser,
        getUser: getUser
    }
});