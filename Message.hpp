
/// The type of message identifies the purpose of a message
enum MessageType
{
    /// Message contains a requested cache line (Receiver: Tile)
    MessageTypeResponseCacheline = 1,
    
    /// Message requests a cacheline from off-tile L2 (Receiver: Tile)
    MessageTypeRequestL2,
    
    /// Message requests a cacheline from memory (Receiver: Main Memory MMU)
    MessageTypeRequestMem
};

/// Might or might not be useful
struct NodeId
{
    union
    {
        int2 tileIdx;
        int someMemoryIdentifier;
    }
};


/// A message can be sent through the network for 
struct Message
{
    /// The type of message
    MessageType type;
    
    /// The id of the sender (tile or Global MMU)
    int2 sender;
    
    /// The id of the receiver (Tile, core block or Global MMU)
    int2 receiver;

    /// The id of the request in the sender that this Message is serving
    int requestId;

    /// The total delay that the current instruction (i.e. memory access) has cost so far
    int totalDelay;

    union
    {
        Address addr;
        int cacheLine[GlobalCfg::CacheLineSize];
    }

    /// Whether this message is broadcasted to all cores of a core block
    bool IsBroadcast() const
    {
        // We simplify our simulation for now in such a way that there is no broadcasting required
        return false;
    }

    /// Whether this message is being sent to another tile (or main memory, if false)
    bool IsReceiverTile() const
    {
        return type != MessageTypeRequestMem;
    }
};