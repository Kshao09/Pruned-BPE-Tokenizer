package se.dsv.waora.deviceinternetinformation;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.Bundle;
import android.support.v7.app.ActionBarActivity;
import android.widget.TextView;


/**
 * <code>ConnectionActivity</code> presents UI for showing if the device
 * is connected to internet.
 *
 * @author Dushant Singh
 */
public class ConnectionActivity extends ActionBarActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initiate view
        TextView connectivityStatus = (TextView) findViewById(R.id.textViewDeviceConnectivity);

        // Get connectivity service.
        ConnectivityManager manager = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);

        // Get active network information
        NetworkInfo activeNetwork = manager.getActiveNetworkInfo();

        // Check if active network is connected.
        boolean isConnected = activeNetwork != null && activeNetwork.isConnectedOrConnecting();
        if (isConnected) {
            // Set status connected
            connectivityStatus.setText(getString(R.string.online));
            connectivityStatus.setTextColor(getResources().getColor(R.color.color_on));

            // Check if connected with wifi
            boolean isWifiOn = activeNetwork.getType() == ConnectivityManager.TYPE_WIFI;
            if (isWifiOn) {
                // Set wifi status on
                TextView wifiTextView = (TextView) findViewById(R.id.textViewWifi);
                wifiTextView.setText(getString(R.string.on));
                wifiTextView.setTextColor(getResources().getColor(R.color.color_on));

            } else {
                // Set mobile data status on.
                TextView mobileDataTextView = (TextView) findViewById(R.id.textViewMobileData);
                mobileDataTextView.setText(getString(R.string.on));
                mobileDataTextView.setTextColor(getResources().getColor(R.color.color_on));
            }
        }
    }
}
