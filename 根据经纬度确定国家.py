import android.location.Address as ad;

public String getAddress(double latitude, double longitude) {
    String cityName = "";
    List<Address> addList = null;
    Geocoder ge = new Geocoder(BoneActivity.this);
    try {
        addList = ge.getFromLocation(latitude,longitude,1);
    } catch(IOException e){
        e.printStackTrace();
    }
    if (addList != null && addList.size()>0){
        for (int i=0;i<addList.size();i++){
             Address ad = addList.get(i);
             cityName += ad.getCountryName()+";"+ad.getLocality();
        }
    }
    return cityName
}