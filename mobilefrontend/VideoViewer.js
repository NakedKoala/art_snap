import React, { Component } from "react";
import {
    View,
    Text,
    StyleSheet,
    TouchableHighlight,
    Linking
} from "react-native";




const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: "flex-start"
    },
    headline: {
        alignSelf: "center",
        fontSize: 18,
        marginTop: 10,
        marginBottom: 30
    },
    videoTile: {
        alignSelf: "center",
        fontSize: 16,
        marginTop: 15
    }
})



export default class VideoOverview extends Component {
    constructor() {
        super();

    }



    openVideo() {
        const url = `https://s3.amazonaws.com/artsnap-userfiles-mobilehub-1207532684/public/testvideo.mp4`;

        Linking.openURL(url).catch(err =>
            console.error("An error occurred opening the link", err)
        );
    }


    render() {


        return (
            <View style={styles.container}>
                <Text style={styles.headline}>Videos</Text>

                <TouchableHighlight
                    underlayColor="rgba(200,200,200,0.6)"
                    onPress={this.openVideo.bind(this)}
                >
                    <Text style={styles.videoTile}>Watch </Text>
                </TouchableHighlight>

            </View>
        );
    }


}