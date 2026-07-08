
function collectWithWildcard(test) {
	test.expect(4);

	var api_server = new Test_ApiServer(function handler(request, callback) {
		var url = request.url;

		switch (url) {
			case '/accounts?username=chariz*':
				let account = new Model_Account({
					username: 'charizard'
				});

				return void callback(null, [
					account.redact()
				]);

			default:
				let error = new Error('Invalid url: ' + url);

				return void callback(error);
		}
	});

	var parameters = {
		username: 'chariz*'
	};

	function handler(error, results) {
		test.equals(error, null);
		test.equals(results.length, 1);

		var account = results[0];

		test.equals(account.get('username'), 'charizard');
		test.equals(account.get('type'), Enum_AccountTypes.MEMBER);

		api_server.destroy();

		test.done();
	}

	Resource_Accounts.collect(parameters, handler);
}

module.exports = {
	collectWithWildcard
};
